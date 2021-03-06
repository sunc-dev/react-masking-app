"""Docstring for the encryption.py module.

encryption module has two main functionalities, (1) dump serialized values into json files 
and (2) load from json files.

This module is used to store serialized keys and encrypted values in json files for the 
privacy reason and enables reading them from json files and deserialize them as well. 

"""

import phe
from phe import paillier
import json 


enc_folder = "encryption_keys"

def keypair_dump(pub, priv, date=None):
    r"""keypair_dump funcion is used to serialize public and private keypair to 
    json web key (jwk) format and dump them into .priv and .pub files for private and 
    public keys respectively

    """
    from datetime import datetime
    if date is None:
        date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    rec_pub = {
        'kty': 'DAJ',
        'alg': 'PAI-GN1',
        'key_ops': ['encrypt'],
        'n': phe.util.int_to_base64(pub.n),
        'kid': 'Paillier public key generated by phe on {}'.format(date)
    }

    rec_priv = {
        'kty': 'DAJ',
        'key_ops': ['decrypt'],
        'p': phe.util.int_to_base64(priv.p),
        'q': phe.util.int_to_base64(priv.q),
        'kid': 'Paillier private key generated by phe on {}'.format(date)
    }

    priv_jwk = json.dumps(rec_priv)
    pub_jwk = json.dumps(rec_pub)

    with open(enc_folder + "/phe_key.priv", "w") as f:
        f.write(priv_jwk + "\n")

    with open(enc_folder + "/phe_key.pub", "w") as f:
        f.write(pub_jwk + "\n")

def keypair_load(pub_file, priv_file):
    r"""keypair_load funcion is used to deserialize public and private keypair from 
    jwk format and returns public and private keys

    Returns
    -------
    Pailliar keys: returns deserialized public and private keys
    """
    with open(pub_file, 'r') as f: 
        pub_jwk = f.read()

    with open(priv_file, 'r') as f: 
        priv_jwk = f.read()
    
    rec_pub = json.loads(pub_jwk)
    rec_priv = json.loads(priv_jwk)

    #quality check
    assert rec_pub['kty'] == "DAJ", "Invalid public key type"
    assert rec_pub['alg'] == "PAI-GN1", "Invalid public key algorithm"
    assert rec_priv['kty'] == "DAJ", "Invalid private key type"

    pub_n = phe.util.base64_to_int(rec_pub['n'])
    pub = paillier.PaillierPublicKey(pub_n)
    priv_p = phe.util.base64_to_int(rec_priv['p'])
    priv_q = phe.util.base64_to_int(rec_priv['q'])
    priv = paillier.PaillierPrivateKey(pub, priv_p, priv_q)
    return pub, priv

def envec_dump_json(pubkey, enc_vals: list, jsonfile, indent=None):
    r"""envec_dump_json funcion is used to serialize a list of encrypted coefficients 
    and dump them into a simple json file. 

    Parameters
    ----------
    pubkey : is the pailliar public key used for encryption
    enc_vals : the list of encrypted coefficients 
    jsonfile: is a string presenting the file name for the json dump

    """

    from phe.util import int_to_base64
    r = {}
    r['public_key'] = {
        'n': int_to_base64(pubkey.n),
    }
    r['values'] = [
        (int_to_base64(x.ciphertext()), x.exponent) for x in enc_vals
    ]

    with open(jsonfile, 'w') as outfile:
      json.dump(r, outfile) 

def envec_load_json(r_json:str):
    r"""envec_load_json funcion is used to deserialize a list of encrypted coefficients 

    Parameters
    ----------
    r_json : is a string presenting the file including the encrypted coefficients 

    Returns
    ----------
    public key and a vector/list of encrypted coefficients 

    """
    
    from phe.util import base64_to_int
    with open(r_json) as json_data:
      r = json.load(json_data)
    r_pubkey = r['public_key']
    r_values = r['values']

    pubkey_d = paillier.PaillierPublicKey(n=base64_to_int(r_pubkey['n']))
    values_d = [
        paillier.EncryptedNumber(pubkey_d, ciphertext=base64_to_int(v[0]), exponent=int(v[1]))
        for v in r_values
    ]
    return pubkey_d, values_d
