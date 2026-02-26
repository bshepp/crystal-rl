#!/usr/bin/env python3
"""Check AFLOW schema for effective mass fields."""
import requests, json, re

r = requests.get('http://aflowlib.duke.edu/API/aflux/?schema,paging(0,1)', timeout=30)
text = r.text.lower()

# Find quoted strings containing 'mass'
matches = re.findall(r'"([^"]*mass[^"]*)"', text)
for m in sorted(set(matches)):
    print('MASS:', m)

print('---')
matches2 = re.findall(r'"([^"]*effective[^"]*)"', text)
for m in sorted(set(matches2)):
    print('EFFECTIVE:', m)

print('---')
# Check for AEL/AGL 
matches3 = re.findall(r'"(ael[^"]*)"', text)
for m in sorted(set(matches3)):
    print('AEL:', m)
    
matches4 = re.findall(r'"(agl[^"]*)"', text)
for m in sorted(set(matches4)):
    print('AGL:', m)
    
print('---')
# Check for 'electron' or 'hole' 
matches5 = re.findall(r'"([^"]*electron[^"]*)"', text)
for m in sorted(set(matches5)):
    if len(m) < 60:
        print('ELECTRON:', m)
