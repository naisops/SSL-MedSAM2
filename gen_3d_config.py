# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:12:13 2025

@author: naisops
"""

import argparse


class Config:
    vendor = 'Vendor_A'
    total_part = 1
    run_part = 0
    n_sampling_on_each_mask = 5
    num_class = 1
    sequence = 'ged4'
    random_mix_method_name = 'random_whole_insert' # random_switch, random_whole_insert, one_by_one_switch
    insert_mod = 'la_to_un' # un_to_la, la_to_un


    @staticmethod
    def parse_args():
        """
        Parse command-line arguments and override default configurations.
        Returns:
            argparse.Namespace: Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Configuration Overrides")
        parser.add_argument("--vendor", type=str, default=Config.vendor, help="vendor type")
        parser.add_argument("--seq", type=str, default=Config.sequence, help="sequence type")
        parser.add_argument("--mix", type=str, default=Config.random_mix_method_name, help="random mixing method name")
        parser.add_argument("--insert_mod", type=str, default=Config.insert_mod, help="whole inserting mod")

        parser.add_argument("--run_part", type=int, default=Config.run_part, help="which part to run")
        parser.add_argument("--total_part", type=int, default=Config.total_part, help="the number of total parts")
        parser.add_argument("--n_sampling", type=int, default=Config.n_sampling_on_each_mask, help="number of random sampling per GT")
        parser.add_argument("--num_class", type=int, default=Config.num_class, help="number of foreground classes")

        # Use parse_known_args() to ignore unrecognized arguments
        args, _ = parser.parse_known_args()
        return args

    @staticmethod
    def get_config():
        """
        Combine default configurations with command-line overrides.
        Returns:
            Config: A Config object with updated values.
        """
        args = Config.parse_args()
        config = Config()

        config.vendor = args.vendor
        config.run_part = args.run_part
        config.n_sampling_on_each_mask = args.n_sampling
        config.num_class = args.num_class
        config.total_part = args.total_part
        config.sequence = args.seq
        config.random_mix_method_name = args.mix
        config.insert_mod = args.insert_mod

        return config
