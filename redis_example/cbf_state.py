import redis, os
from configparser import ConfigParser # SafeConfigParser has been deprecated

import logging

def parse_config_file(config_file = ''):
    """
    Parse an config file into a dictionary. No checking done at all.
    :param config_file: the config file to process
    :param required_sections: sections that MUST be included
    :return: a dictionary containing the configuration
    """
    parser = ConfigParser()
    files = parser.read(config_file)
    if len(files) == 0:
        raise IOError('Could not read the config file, %s' % config_file)
    
    config = {}

    for section in parser.sections():
        config[section] = {}
        for items in parser.items(section):
            config[section][items[0]] = items[1]
    
    return config


class CbfState(object):
    """
    This is the first attempt at a 'CbfState'
    - Using the default parameters passed on to the redis.Redis server instance
    """
    def __init__(self,
                 descriptor=None,
                 config_file=None,
                 host='localhost',
                 port=6379,
                 db=0,
                 log_level='DEBUG'):
        """
        :param config_file:
            The configuration file used to create the instrument
        :type config_file:
            str
        """
        self.descriptor = descriptor
        if self.descriptor is None:
            self.descriptor = config_file.strip().replace(' ', '_').lower()

        self.logger = logging.getLogger(self.descriptor)
        logging.basicConfig()
        # self.logger.setLevel(logging.DEBUG)
        
        # Check if the config-file is legit
        abs_path = os.path.abspath(config_file)
        if not os.path.exists(abs_path):
            # Problem
            errmsg = 'Config-file {} is not valid'.format(config_file)
            self.logger.error(errmsg)
            raise ValueError(errmsg)
        # else: Continue
        self.config_dict = parse_config_file(config_file)

        self.redis_server = redis.Redis(host=host, port=port, db=db)

    
    def initialise(self):
        """
        Populate the Redis Server with the config-file data
        - Will need to use a Hashed-set
            - redis_server.hset(key, value, number)
        """
        # self.config_dict is a dictionary of dictionaries
        # - list(config.dict.keys()) returns an iterable list of keys


        with self.redis_server.pipeline() as pipe:
            # To queue the transactions and send together
            for main_key, value_dict in self.config_dict.items():
                # pipe.hmset(main_key, value_dict) - DEPRECATED
                # pipe.hset(main_key, mapping=value_dict) - Mapping isn't working :(
                for sub_key, sub_value in value_dict.items():
                    pipe.hset(name=main_key, key=sub_key, value=sub_value)

            pipe.execute()


    def get_keys(self):
        """
        Abstracting the redis_server.keys() command
        """
        return self.redis_server.keys()

    
    def get_value_dict(self, key=None):
        """
        Abstracting redis_server.hgetall(key)        
        """
        if key is None:
            self.logger.error('Please specify a key')
            return None
        
        return self.redis_server.hgetall(key)


    def clear_database(self):
        """
        To clear all data in the database
        """
        self.redis_server.flushdb()



