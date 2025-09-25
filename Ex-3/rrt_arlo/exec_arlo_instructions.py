import move_arlo

def exec_instructions(arlo_instructions):
    for instruction in arlo_instructions:
        getattr(move_arlo, instruction[0])(instruction[1])

if __name__ == '__main__':
    instructions = [['turn', (True, 35.1)], ['forward', 0.71], ['turn', (True, 58.42)], ['forward', 0.42], ['turn', (True, 47.92)], ['forward', 0.48], ['turn', (True, 48.97)], ['forward', 0.53], ['turn', (False, 46.83)], ['forward', 0.48]]
    exec_instructions(instructions)