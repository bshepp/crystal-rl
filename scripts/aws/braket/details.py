#!/usr/bin/env python3
"""Get detailed Braket QPU specs for Ankaa-3 and Aquila."""
import subprocess, json, os
os.environ['PYTHONUTF8'] = '1'

def get_device_details(arn, region):
    r = subprocess.run(
        ['aws', 'braket', 'get-device', '--device-arn', arn, '--region', region, '--output', 'json'],
        capture_output=True, encoding='utf-8', errors='replace'
    )
    d = json.loads(r.stdout)
    caps = json.loads(d.get('deviceCapabilities', '{}'))
    return d, caps

print("=" * 60)
print("RIGETTI ANKAA-3 (us-west-1)")
print("=" * 60)
d, caps = get_device_details('arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3', 'us-west-1')
paradigm = caps.get('paradigm', {})
print(f"Qubits: {paradigm.get('qubitCount')}")
print(f"Native gates: {paradigm.get('nativeGateSet')}")
conn = paradigm.get('connectivity', {})
print(f"Fully connected: {conn.get('fullyConnected')}")
# Count edges in connectivity graph
graph = conn.get('connectivityGraph', {})
total_edges = sum(len(v) for v in graph.values()) // 2
print(f"Connectivity edges: {total_edges}")
print(f"Topology type: lattice/grid (superconducting)")

# Provider specs (fidelities etc)
provider = caps.get('provider', {})
if provider:
    specs = provider.get('specs', {})
    for category, values in specs.items():
        print(f"\n  {category}:")
        if isinstance(values, dict):
            for k, v in list(values.items())[:5]:
                print(f"    {k}: {v}")
            if len(values) > 5:
                print(f"    ... ({len(values)} total entries)")

# Cost
service = caps.get('service', {})
cost = service.get('deviceCost', {})
print(f"\nCost: {cost}")
shot_range = caps.get('action', {})
for action_name, action_val in shot_range.items():
    if 'maximumShots' in str(action_val):
        if isinstance(action_val, dict):
            print(f"Shots: min={action_val.get('minimumShots', '?')}, max={action_val.get('maximumShots', '?')}")

print("\n" + "=" * 60)
print("QUERA AQUILA (us-east-1)")
print("=" * 60)
d2, caps2 = get_device_details('arn:aws:braket:us-east-1::device/qpu/quera/Aquila', 'us-east-1')
paradigm2 = caps2.get('paradigm', {})
print(f"Qubits (atoms): {paradigm2.get('qubitCount')}")
print(f"Lattice area: {paradigm2.get('lattice', {}).get('area', 'N/A')}")
print(f"Geometry: {paradigm2.get('lattice', {}).get('geometry', 'N/A')}")

# What kind of paradigm is this?
print(f"Paradigm keys: {list(paradigm2.keys())}")

# Provider
provider2 = caps2.get('provider', {})
if provider2:
    for k, v in provider2.items():
        print(f"  {k}: {str(v)[:200]}")

cost2 = caps2.get('service', {}).get('deviceCost', {})
print(f"\nCost: {cost2}")

# Actions
print(f"\nSupported actions: {list(caps2.get('action', {}).keys())}")
for action_name, action_val in caps2.get('action', {}).items():
    if isinstance(action_val, dict):
        print(f"  {action_name}: version={action_val.get('version','?')}")

# Also check simulators pricing
print("\n" + "=" * 60)
print("SIMULATOR PRICING")
print("=" * 60)
for name, arn, region in [
    ('SV1', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1', 'us-east-1'),
    ('TN1', 'arn:aws:braket:::device/quantum-simulator/amazon/tn1', 'us-east-1'),
    ('DM1', 'arn:aws:braket:::device/quantum-simulator/amazon/dm1', 'us-east-1'),
]:
    _, c = get_device_details(arn, region)
    cost = c.get('service', {}).get('deviceCost', {})
    qubits = c.get('paradigm', {}).get('qubitCount', '?')
    print(f"{name}: {qubits} qubits, cost={cost}")
