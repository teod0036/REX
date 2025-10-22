import move_arlo

def all(arlo_instructions):
    for i in range(len(arlo_instructions)):
        next(arlo_instructons=arlo_instructions)

def next(arlo_instructons, rm=True):
    distance_driven = getattr(move_arlo, arlo_instructons[0][0])(arlo_instructons[0][1])
    print(distance_driven)
    if distance_driven != 0:
        del arlo_instructons[1:]
        if distance_driven < 0:
            arlo_instructons.append(["turn", (False, 90)])
            arlo_instructons.append(["forward", (0.5)])
            arlo_instructons.append(["turn", (True, 90)])
            arlo_instructons.append(["forward", (0.5)])
            arlo_instructons.append(["turn", (True, 90)])
            arlo_instructons.append(["forward", (0.5)])
            arlo_instructons.append(["turn", (False, 90)])
            distance_driven = distance_driven * -1
        else:
            arlo_instructons.append(["turn", (True, 90)])
            arlo_instructons.append(["forward", (0.5)])
            arlo_instructons.append(["turn", (False, 90)])
            arlo_instructons.append(["forward", (0.5)])
            arlo_instructons.append(["turn", (False, 90)])
            arlo_instructons.append(["forward", (0.5)])
            arlo_instructons.append(["turn", (True, 90)])

        arlo_instructons[0][1] = distance_driven

    if rm:
        del arlo_instructons[0]

if __name__ == "__main__":
    instructions = [["forward", (3)], ["forward", (0.5)]]
    next(instructions, rm=False)
    all(instructions)