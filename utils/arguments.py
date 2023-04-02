import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str,\
            default='SLP',\
            help='[IF, \nSLP, MLP, \nCNN, LSTM, Transformer]')
    parser.add_argument('--activation_type', type=str,\
            default='sign')
    parser.add_argument('--input_dim', type=int,\
            default=784)
    parser.add_argument('--hidden_dim', type=int,\
            default=32)
    parser.add_argument('--output_dim', type=int,\
            default=10)

    parser.add_argument('--data_name', type=str,\
            default='MNIST')
    
    parser.add_argument('--epoch', type=int,\
            default=100)
    parser.add_argument('--learning_rate', type=float,\
            default=1e-3)
    parser.add_argument('--batch_size', type=int,\
            default=512)


    args = parser.parse_args()

    return args

