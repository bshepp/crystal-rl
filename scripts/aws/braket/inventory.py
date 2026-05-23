#!/usr/bin/env python3
"""Get Amazon Braket device inventory."""
import subprocess, json, os, sys
os.environ['PYTHONUTF8'] = '1'

devices = [
    ('arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3', 'us-west-1'),
    ('arn:aws:braket:us-east-1::device/qpu/quera/Aquila', 'us-east-1'),
    ('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1', 'us-east-1'),
    ('arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1', 'us-east-1'),
]

simulators = [
    ('arn:aws:braket:::device/quantum-simulator/amazon/sv1', 'us-east-1'),
    ('arn:aws:braket:::device/quantum-simulator/amazon/tn1', 'us-east-1'),
    ('arn:aws:braket:::device/quantum-simulator/amazon/dm1', 'us-east-1'),
]

print(f"{'Name':>14s} | {'Provider':>10s} | {'Status':>8s} | {'Type':>5s} | {'Qubits':>8s} | Region")
print("-" * 80)

for arn, region in devices + simulators:
    r = subprocess.run(
        ['aws', 'braket', 'get-device', '--device-arn', arn, '--region', region, '--output', 'json'],
        capture_output=True, encoding='utf-8', errors='replace'
    )
    if r.returncode != 0:
        short = arn.split('/')[-1]
        print(f"{short:>14s} | {'N/A':>10s} | {'ERROR':>8s} | {'?':>5s} | {'?':>8s} | {region}")
        continue
    try:
        d = json.loads(r.stdout)
        caps = json.loads(d.get('deviceCapabilities', '{}'))
        paradigm = caps.get('paradigm', {})
        qubits = paradigm.get('qubitCount', '?')
        name = d['deviceName']
        provider = d['providerName']
        status = d['deviceStatus']
        dtype = d['deviceType']
        print(f"{name:>14s} | {provider:>10s} | {status:>8s} | {dtype:>5s} | {str(qubits):>8s} | {region}")

        # Extra details for online QPUs
        if status == 'ONLINE' and dtype == 'QPU':
            connectivity = paradigm.get('connectivity', {})
            native_gates = paradigm.get('nativeGateSet', [])
            print(f"               Connectivity: {connectivity.get('fullyConnected', 'partial')}, type={connectivity.get('connectivityGraph', 'N/A')[:80]}")
            print(f"               Native gates: {native_gates}")
            # Fidelities
            perf = caps.get('provider', {})
            if 'specs' in perf:
                specs = perf['specs']
                for spec_name, spec_vals in list(specs.items())[:3]:
                    print(f"               {spec_name}: {str(spec_vals)[:120]}")
            # Pricing
            cost = caps.get('service', {}).get('deviceCost', {})
            if cost:
                print(f"               Cost: {cost}")
    except Exception as e:
        short = arn.split('/')[-1]
        print(f"{short:>14s} | parse error: {e}")
