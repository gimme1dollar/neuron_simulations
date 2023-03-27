from utils.arguments import get_args

def main(args):
    algorithm = args.algorithm
    if algorithm == 'IF':
        from algorithm.Integrate_And_Fire.IntegrateAndFire import main as experiment
        arguments = {
                'c': 10,
                'r': 10,
                'v_0': 10, 'v_th': 10, 'v_sp': 10,
            }
    if algorithm == 'SLP':
        from algorithm.Perceptron.Single_layer_perceptron import main as experiment
        arguments = {'test': None}

    experiment(args, **arguments)
    return

if __name__ == '__main__':
    main(get_args())
