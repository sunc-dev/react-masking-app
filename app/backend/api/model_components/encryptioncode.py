
import sys
import os
from os import walk, path
import pandas as pd 
from datetime import datetime
import phe
from phe import paillier
import json 
import random 
from obfuscation import obfuscation
from encryption import keypair_dump, keypair_load, envec_dump_json, envec_load_json

def code_creation(enc_folder):
    #enc_folder = "Encryption_keys"
    #using previously generated coefficients 
    if path.exists(enc_folder+'/keys.json'):
        pb, pv = keypair_load(enc_folder+"/phe_key.pub", enc_folder+"/phe_key.priv")
        coeff = [pv.decrypt(x) for x in envec_load_json(enc_folder+'/keys.json')[1]]   

    #randomly generate a new set of coefficients 
    else: 
        if not path.exists(enc_folder):
            os.mkdir(enc_folder)
        time = datetime.now()
        coeff = [random.randint(1, 3)] + random.sample(range(2, time.month), 3)
        pb_key, pv_key = paillier.generate_paillier_keypair()
        keypair_dump(pb_key, pv_key)

        enc_coeff = [pb_key.encrypt(x) for x in coeff]
        envec_dump_json(pb_key, enc_coeff, enc_folder+'/keys.json')

    return coeff