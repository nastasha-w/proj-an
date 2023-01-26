#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
not using the standard .format stuff, since this fills in a bash
scripts and those are just full of ${varname} stuff

Parameters:
-----------
a method specification, e.g. --firemaps_frontera_seq

for --firemaps_frontera_seq: (fire_maps.py calls with sequential indices)
    --PYFILL_TIME_HHMMSS=HH:MM:SS
    --PYFILL_PARTITION=<flex OR small>
    --PYFILL_JOBNAME=<job name>
    --start=<first index, int>
    --step=<number of tasks per node, int>
    --last=<last index, int>

    generates a set of slurm scripts with tasks divided as evenly as possible.
    Each calls fire_maps.py with a number of indices, so that all indices from
    start to last are called, with step or (step - 1) on each node. No jobs 
    with more than one node are created.
'''

import os
import stat
import sys

import numpy as np

# where the templates are and the sbatch files go
sdir = '/work2/08466/tg877653/frontera/slurm/'

def fillin(templatefilen, outfilen, **kwargs):
    '''
    reads in templatefilen, replaces instances of each kwarg key
    with the corresponding value, writes out the result to outfilen.
    '''
    #print(templatefilen)
    with open(templatefilen, 'r') as f:
        template = f.read()
    out = template
    for key in kwargs:
        val = kwargs[key]
        out = out.replace(key, str(val))
    with open(outfilen, 'w') as f_out:
        f_out.write(out)
    # make executable (for me)
    os.chmod(outfilen, '0700')

def fillin_firemaps_frontera_seqinds(**kwargs):
    '''
    fill in template_slurm_frontera_autolauncher.sh, checking the key
    values
    '''
    templatefilen = sdir + 'template_slurm_frontera_autolauncher.sh'
    # standard format to document, jobname for batch queue submission
    outfilen = sdir + 'slurm_firemaps_{st}_to_{ls}_{jobname}.sh'
    defaults = {'PYFILL_TIME_HHMMSS': '01:00:00',
                'PYFILL_PARTITION': 'flex'}
    keys_req = ['PYFILL_JOBNAME',
                'PYFILL_NTASKS',
                'PYFILL_FMIND_START']
    keys_opt = list(defaults.keys())
    kwargs_next = defaults.copy()
    for key in kwargs:
        if (key not in keys_req) and (key not in keys_opt):
            msg = 'skipping key {}: not an option for this template'
            print(msg.format(key))
            continue
        kwargs_next.update({key: kwargs[key]})
        if key in keys_req:
            keys_req.remove(key)
    if len(keys_req) > 0:
        print('required keys missing: {}'.format(keys_req))
    
    last = kwargs_next['PYFILL_FMIND_START'] + kwargs_next['PYFILL_NTASKS'] - 1
    outfilen = outfilen.format(st=kwargs_next['PYFILL_FMIND_START'],
                               ls=last,
                               jobname=kwargs_next['PYFILL_JOBNAME'])
    
    fillin(templatefilen, outfilen, **kwargs_next)

def fillset_firemaps_frontera_seqinds(**kwargs):
    keys_req = ['start', 'step', 'last']
    for key in keys_req:
        if key not in kwargs:
            raise ValueError('Argument {} missing'.format(key))
    start = int(kwargs['start'])
    step = int(kwargs['step'])
    last = int(kwargs['last'])
    for key in keys_req:
        del kwargs[key]
    _kwargs_next = kwargs.copy()
    numfiles = (last - start) // step + 1
    if (last - start + 1) % step == 0:
        ntaskss = [step] * numfiles
        starts = [start + step * i for i in range(numfiles)]
    else:
        numadd = (last - start + 1) % step
        ntaskss = [step if i < numadd else step - 1 for i in range(numfiles)]
        starts = [start + sum(ntaskss[:i]) for i in range(numfiles)]
    for ntasks, start in zip(ntaskss, starts):
        kwargs_next = _kwargs_next.copy()
        kwargs_next['PYFILL_NTASKS'] = ntasks
        kwargs_next['PYFILL_FMIND_START'] = start
        fillin_firemaps_frontera_seqinds(**kwargs_next)

if __name__ == '__main__':
    args = sys.argv[1:]
    methodargs = ['--firemaps_frontera_seq']
    methodset = False
    kwargs = {}
    for arg in args:
        if arg in methodargs:
            if methodset:
                raise ValueError('multiple template options specified')
            methodset = True
            continue
        elif arg.startswith('--'):
            arg = arg[2:]
            key, val = arg.split('=')
        else:
            key, val = arg.split('=')
        kwargs.update({key: val})

    if '--firemaps_frontera_seq' in args:
        fillset_firemaps_frontera_seqinds(**kwargs)
    else: 
        msg = 'No known template specified; options are: {}'
        raise ValueError(msg.format(methodargs))