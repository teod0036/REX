import move_arlo

def all(arlo_instructions, rm=True):
    for instruction in arlo_instructions:
        getattr(move_arlo, instruction[0])(instruction[1])
    if rm:
        del arlo_instructions[:]

def next(arlo_instructons, rm=True):
    getattr(move_arlo, arlo_instructons[0][0])(arlo_instructons[0][1])
    if rm:
        del arlo_instructons[0]