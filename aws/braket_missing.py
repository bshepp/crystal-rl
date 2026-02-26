#!/usr/bin/env python3
"""Get details on QPUs we missed: AQT IBEX Q1, IonQ Forte Enterprise 1, IQM Garnet, IQM Emerald."""
import subprocess, json, os
os.environ['PYTHONUTF8'] = '1'

# Try various ARN patterns and regions for each device
candidates = [
    # AQT IBEX Q1 - likely eu-central-1
    ('arn:aws:braket:eu-central-1::device/qpu/aqt/IBEX_Q1', 'eu-central-1'),
    ('arn:aws:braket:eu-central-1::device/qpu/aqt/Ibex-Q1', 'eu-central-1'),
    ('arn:aws:braket:eu-central-1::device/qpu/aqt/ibex-q1', 'eu-central-1'),
    ('arn:aws:braket:us-east-1::device/qpu/aqt/IBEX_Q1', 'us-east-1'),
    # IonQ Forte Enterprise 1
    ('arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1', 'us-east-1'),
    ('arn:aws:braket:us-east-1::device/qpu/ionq/Forte_Enterprise_1', 'us-east-1'),
    ('arn:aws:braket:us-east-1::device/qpu/ionq/ForteEnterprise-1', 'us-east-1'),
    # IQM Garnet - try multiple regions
    ('arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet', 'eu-north-1'),
    ('arn:aws:braket:eu-west-2::device/qpu/iqm/Garnet', 'eu-west-2'),
    ('arn:aws:braket:us-east-1::device/qpu/iqm/Garnet', 'us-east-1'),
    ('arn:aws:braket:us-west-1::device/qpu/iqm/Garnet', 'us-west-1'),
    ('arn:aws:braket:eu-central-1::device/qpu/iqm/Garnet', 'eu-central-1'),
    # IQM Emerald
    ('arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald', 'eu-north-1'),
    ('arn:aws:braket:eu-west-2::device/qpu/iqm/Emerald', 'eu-west-2'),
    ('arn:aws:braket:eu-central-1::device/qpu/iqm/Emerald', 'eu-central-1'),
]

found = set()
for arn, region in candidates:
    short_name = arn.split('/')[-1]
    if short_name in found:
        continue
    r = subprocess.run(
        ['aws', 'braket', 'get-device', '--device-arn', arn, '--region', region, '--output', 'json'],
        capture_output=True, encoding='utf-8', errors='replace'
    )
    if r.returncode != 0:
        continue
    try:
        d = json.loads(r.stdout)
        caps = json.loads(d.get('deviceCapabilities', '{}'))
        paradigm = caps.get('paradigm', {})
        qubits = paradigm.get('qubitCount', '?')
        conn = paradigm.get('connectivity', {})
        gates = paradigm.get('nativeGateSet', [])
        cost = caps.get('service', {}).get('deviceCost', {})
        found.add(short_name)
        
        print(f"FOUND: {d['deviceName']} ({d['providerName']})")
        print(f"  ARN: {arn}")
        print(f"  Region: {region}")
        print(f"  Status: {d['deviceStatus']}")
        print(f"  Qubits: {qubits}")
        print(f"  Fully connected: {conn.get('fullyConnected', 'N/A')}")
        print(f"  Native gates: {gates}")
        print(f"  Cost: {cost}")
        
        # Supported actions
        actions = list(caps.get('action', {}).keys())
        print(f"  Actions: {actions}")
        
        # Provider specs summary
        provider = caps.get('provider', {})
        specs = provider.get('specs', {})
        if specs:
            for cat, vals in list(specs.items())[:2]:
                print(f"  Specs/{cat}: {str(vals)[:200]}")
        print()
    except Exception as e:
        pass

if not found:
    print("No new devices found with tried ARN patterns.")
    print("Let's try search-devices instead...")
    for region in ['us-east-1', 'us-west-1', 'us-west-2', 'eu-central-1', 'eu-north-1', 'eu-west-2']:
        r = subprocess.run(
            ['aws', 'braket', 'search-devices', '--region', region, '--output', 'json'],
            capture_output=True, encoding='utf-8', errors='replace'
        )
        if r.returncode == 0:
            devices = json.loads(r.stdout).get('devices', [])
            for dev in devices:
                if dev.get('deviceType') == 'QPU':
                    print(f"  {region}: {dev['deviceName']} ({dev['providerName']}) - {dev['deviceStatus']} - {dev['deviceArn']}")
