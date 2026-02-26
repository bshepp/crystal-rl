#!/usr/bin/env python3
"""Find AQT devices across all Braket regions."""
import subprocess, json, os
os.environ['PYTHONUTF8'] = '1'

regions = ['us-east-1', 'us-west-1', 'us-west-2', 'eu-central-1', 'eu-north-1', 
           'eu-west-1', 'eu-west-2', 'ap-northeast-1', 'ap-southeast-1', 'ap-south-1']

for region in regions:
    r = subprocess.run(
        ['aws', 'braket', 'search-devices', '--region', region, '--output', 'json'],
        capture_output=True, encoding='utf-8', errors='replace'
    )
    if r.returncode != 0:
        continue
    try:
        devs = json.loads(r.stdout).get('devices', [])
        qpus = [d for d in devs if d.get('deviceType') == 'QPU']
        if qpus:
            for d in qpus:
                print(f"{region:>16s}: {d['deviceName']:>25s} ({d['providerName']}) - {d['deviceStatus']} - {d['deviceArn']}")
    except:
        pass
