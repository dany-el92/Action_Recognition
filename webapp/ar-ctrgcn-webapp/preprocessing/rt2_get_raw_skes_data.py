import numpy as np


def get_raw_bodies_data(ske_name, ske_data):
    actors = ['actor1', 'actor2']
    num_frames = ske_data['num_frames']
    num_bodies = ske_data['num_actors']
    num_joints = 17
    #frames_drop = []
    bodies_data = dict()

    baseid = 'test_id_'
    bodyIDs = [baseid + '1', baseid + '2']

    data = {}

    final_num_frames = []
    
    for b in range(num_bodies):
        a = actors[b]

        actor_frames = []
        valid_frames = 0

        for f in range(num_frames):

            if ske_data['num_actors'] > 1:
                if not len(ske_data['actor1']['joints'][f]) > 0 and not len(ske_data['actor2']['joints'][f]) > 0:
                    #frames_drop.append(f)
                    continue
            elif ske_data['num_actors'] > 0:
                if not len(ske_data['actor1']['joints'][f]) > 0:
                    #frames_drop.append(f)
                    continue

            valid_frames += 1

            actor_frames.append(ske_data[a]['joints'][f])

        data[a] = {
            'joints': np.array(actor_frames),
            'interval': [*range(valid_frames)]
        }

        final_num_frames.append(valid_frames)

    return {'name': ske_name, 'data': data, 'num_frames': final_num_frames}

def get_raw_skes_data(raw_data):
    ske_name = 'test_skeleton'
    success = False
    raw_skes_data = []
    bodies_data = get_raw_bodies_data(ske_name, raw_data)
    if len(bodies_data['num_frames']) > 0:
        success = True

    #raw_skes_data.append(bodies_data)

    return bodies_data, success

if __name__ == '__main__':
    pass