```bash
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA.
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA.
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs.
python main.py --cuda --epochs 6 --model Transformer --lr 5
                                           # Train a Transformer model on Wikitext-2 with CUDA.
python main.py --cuda --model Transformer --lr 5
                                           # Train a Transformer model on Wikitext-2 with CUDA for 40 epochs.
python generate.py                         # Generate samples from the default model checkpoint.
```

args.bptt - 35, so basically the memory is probably only this scale. 

parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')


1. model_lstm.pt 1774MB 86%
    `python main.py --cuda --epochs 6 --save model_lstm.pt > log_lstm.txt`
    train loss 4.64 valid loss 5.01 test loss 4.93
2. model_lstm_tied.pt 1748MBGPU 85% (slight less weights)
    `python main.py --cuda --epochs 6 --tied --save model_lstm_tied.pt > log_lstm_tied.txt`
    train loss 4.64 valid loss 4.92 test loss 4.85
3. model_lstm_tied_full.pt 1774MB 86%
    `python main.py --cuda --save model_lstm_full.pt > log_lstm_full.txt`
    train loss 3.99 valid loss 4.74 test loss 4.69
4. model_transfromer.pt 1608MB 68%
    `python main.py --cuda --epochs 6 --model Transformer --lr 5 --save model_transformer.pt > log_transformer.txt`
    train loss 4.81 valid loss 5.37 test loss 5.27
5. model_transfromer_full.pt 1608MB 68%
    `python main.py --cuda --model Transformer --lr 5 --save model_transformer_full.pt > log_transformer_full.txt`
    train loss 4.36 valid loss 5.12 test loss 5.05

6. model_rnn_tanh.pt
    `python main.py --cuda --epochs 6 --model RNN_TANH --save model_rnn_tanh.pt > log_rnn_tanh.txt`
    train loss 6.59 valid loss 6.46 test loss 6.39
7. model_rnn_tanh_full.pt
    `python main.py --cuda --model RNN_TANH --save model_rnn_tanh_full.pt > log_rnn_tanh_full.txt`
    train loss 6.20 valid loss 6.16 test loss 6.09

8. model_stablernn_tanh.pt
    `python main.py --cuda --epochs 6 --model StableRNN_TANH --save model_stablernn_tanh.pt > log_stablernn_tanh.txt`
    train loss 6.95 valid loss 6.95 test loss 6.88
9. model_stablernn_tanh_full.pt
    `python main.py --cuda --model StableRNN_TANH --save model_stablernn_tanh_full.pt > log_stablernn_tanh_full.txt`
    train loss 6.99 valid loss 6.86 test loss 6.80



Weighted loss
1. model_WL_lstm.pt 1954MB 94% running
    `python main_WL.py --cuda --epochs 6 --save model_WL_lstm.pt > log_WL_lstm.txt`
    train loss valid loss  test loss 
2. model_WL_transformer.pt 1788MB 91% running
    `python main_WL.py --cuda --epochs 6 --model Transformer --lr 5 --save model_WL_transformer.pt > log_WL_transformer.txt`
    train loss valid loss  test loss 