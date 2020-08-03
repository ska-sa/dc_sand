import asyncio, aiokatcp, logging, time









class DeviceServer(aiokatcp.DeviceServer):
    VERSION = 'cbf-product-controller-0.1'
    # BUILD_STATE = "cbf-product-controller-0.1"

    def __init__(self, host: str, port: int,
                 master_controller: aiokatcp.Client,
                 subarray_product_id: str,
                 
                 batch_role: str,
                 interface_mode: bool,
                 localhost: bool,
                #  image_resolver_factory: scheduler.ImageResolverFactory,
                 s3_config: dict,
                 sched: Optional[scheduler.Scheduler] = None,
                 graph_dir: str = None,
                 dashboard_url: str = None,
                 prometheus_registry: CollectorRegistry = REGISTRY,
                 shutdown_delay: float = 10.0) -> None:
        self.sched = sched
        self.subarray_product_id = subarray_product_id
        self.interface_mode = interface_mode
        self.localhost = localhost
        self.master_controller = master_controller
        self.product: Optional[CBFSubarrayProductBase] = None
        self.shutdown_delay = shutdown_delay

        super().__init__(host, port)
        # setup sensors (note: SDPProductController adds other sensors)
        self.sensors.add(Sensor(DeviceStatus, "device-status",
                                "Devices status of the subarray product controller",
                                default=DeviceStatus.OK,
                                status_func=device_status_to_sensor_status))
        
    async def start(self) -> None:
        # await self.master_controller.wait_connected()
        # await self._consul_register()
        await super().start()

    async def on_stop(self) -> None:
        # await self._consul_deregister()
        # self._prometheus_watcher.close()
        if self.product is not None and self.product.state != ProductState.DEAD:
            logger.warning('Product controller interrupted - deconfiguring running product')
            try:
                await self.product.deconfigure(force=True)
            except Exception:
                logger.warning('Failed to deconfigure product %s during shutdown', exc_info=True)
        # self.master_controller.close()
        # await self.master_controller.wait_closed()

    async def configure_product(self, name: str, config: dict) -> None:
        """Configure a subarray product in response to a request.

        Raises
        ------
        FailReply
            if a configure/deconfigure is in progress
        FailReply
            If any of the following occur
            - The specified subarray product id already exists, but the config
              differs from that specified
            - If docker python libraries are not installed and we are not using interface mode
            - There are insufficient resources to launch
            - A docker image could not be found
            - If one or more nodes fail to launch (e.g. container not found)
            - If one or more nodes fail to become alive
            - If we fail to establish katcp connection to all nodes requiring them.

        Returns
        -------
        str
            Final name of the subarray-product.
        """

        def dead_callback(product):
            if self.shutdown_delay > 0:
                logger.info('Sleeping %.1f seconds to give time for final Prometheus scrapes',
                            self.shutdown_delay)
                asyncio.get_event_loop().call_later(self.shutdown_delay, self.halt, False)
            else:
                self.halt(False)

        logger.debug('config is %s', json.dumps(config, indent=2, sort_keys=True))
        logger.info("Launching subarray product.")

        image_tag = config['config'].get('image_tag')
        if image_tag is not None:
            resolver_factory_args = dict(tag=image_tag)
        else:
            resolver_factory_args = {}
        resolver = Resolver(
            self.image_resolver_factory(**resolver_factory_args),
            scheduler.TaskIDAllocator(name + '-'),
            self.sched.http_url if self.sched else '',
            config['config'].get('service_overrides', {}),
            self.s3_config,
            self.localhost)

        # create graph object and build physical graph from specified resources
        product_cls: Type[CBFSubarrayProductBase]
        # For now, test the product_controller in Interface Mode
        product_cls = CBFSubarrayProductInterface
        product = product_cls(self.sched, config, resolver, name, self)
        if self.graph_dir is not None:
            product.write_graphs(self.graph_dir)
        self.product = product   # Prevents another attempt to configure
        self.product.dead_callbacks.append(dead_callback)
        try:
            await product.configure()
        except Exception:
            self.product = None
            raise

    async def request_product_configure(self, ctx, name: str, config: str) -> None:
        """Configure a CBF Subarray product instance.

        Parameters
        ----------
        name : str
            Name of the subarray product.
        config : str
            A JSON-encoded dictionary of configuration data.
        """
        # TODO: remove name - it is already a command-line argument
        logger.info("?product-configure called with: %s", ctx.req)

        if self.product is not None:
            raise FailReply('Already configured or configuring')
        try:
            config_dict = load_json_dict(config)
            product_config.validate(config_dict)
            config_dict = product_config.normalise(config_dict)
        except product_config.SensorFailure as exc:
            retmsg = f"Error retrieving sensor data from CAM: {exc}"
            logger.error(retmsg)
            raise FailReply(retmsg) from exc
        except (ValueError, jsonschema.ValidationError) as exc:
            retmsg = f"Failed to process config: {exc}"
            logger.error(retmsg)
            raise FailReply(retmsg) from exc

        await self.configure_product(name, config_dict)

    def _get_product(self) -> CBFSubarrayProductBase:
        """Check that self.product exists (i.e. ?product-configure has been called).

        If it has not, raises a :exc:`FailReply`.
        """
        if self.product is None:
            raise FailReply('?product-configure has not been called yet. '
                            'It must be called before other requests.')
        return self.product

    async def request_product_deconfigure(self, ctx, force: bool = False) -> None:
        """Deconfigure the product and shut down the server."""
        await self._get_product().deconfigure(force=force)

    async def request_capture_init(self, ctx, capture_block_id: str,
                                   override_dict_json: str = '{}') -> None:
        """Request capture of the specified subarray product to start.

        Parameters
        ----------
        capture_block_id : str
            The capture block ID for the new capture block.
        override_dict_json : str, optional
            Configuration dictionary to merge with the subarray config.
        """
        product = self._get_product()
        try:
            overrides = load_json_dict(override_dict_json)
        except ValueError as error:
            retmsg = f'Override {override_dict_json} is not a valid JSON dict: {error}'
            logger.error(retmsg)
            raise FailReply(retmsg) from error

        config = product_config.override(product.config, overrides)
        # Re-validate, since the override may have broken it
        try:
            product_config.validate(config)
        except (ValueError, jsonschema.ValidationError) as error:
            retmsg = f"Overrides make the config invalid: {error}"
            logger.error(retmsg)
            raise FailReply(retmsg) from error

        config = product_config.normalise(config)
        try:
            product_config.validate_capture_block(product.config, config)
        except ValueError as error:
            retmsg = f"Invalid config override: {error}"
            logger.error(retmsg)
            raise FailReply(retmsg) from error

        await product.capture_init(capture_block_id, config)

    async def request_telstate_endpoint(self, ctx) -> str:
        """Returns the endpoint for the telescope state repository.

        Returns
        -------
        endpoint : str
        """
        return self._get_product().telstate_endpoint

    async def request_capture_status(self, ctx) -> ProductState:
        """Returns the status of the subarray product.

        Returns
        -------
        state : str
        """
        return self._get_product().state

    async def request_capture_done(self, ctx) -> str:
        """Halts the current capture block.

        Returns
        -------
        cbid : str
            Capture-block ID that was stopped
        """
        cbid = await self._get_product().capture_done()
        return cbid
