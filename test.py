from __future__ import print_function
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.network import net
from data import get_eval_set
from utils import *
from torchvision.transforms import Resize
import time
parser = argparse.ArgumentParser(description='PyTorch USHUIR')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='/home/wr/03_restore/DATA/test_u190/test')
parser.add_argument('--label_test', type=str, default='/home/wr/03_restore/DATA/test_u190/test')
parser.add_argument('--model', default='weights/epoch_50.pth', help='Pretrained base model')
# parser.add_argument('--output_folder', type=str, default='results/predict/')
parser.add_argument('--output_folder', type=str, default='output/test_u190/')
opt = parser.parse_args()

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
test_set = get_eval_set(opt.data_test, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')

model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')


def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    times = []

    for batch in testing_data_loader:
        with torch.no_grad():
            input, label, name = batch[0], batch[1], batch[2]
            print(name)
        input = input.cuda()
        batch, channel, oh, ow = input.size()
        h = int(round(oh / 2) * 2)
        w = int(round(ow / 2) * 2)
        torch_res = Resize([h, w])
        input = torch_res(input)
        with torch.no_grad():
            s = time.time()
            j_out, t_out = model(input)
            times.append(time.time() - s)
            a_out = get_A(input).cuda()
            torch_resize = Resize([input.shape[2], input.shape[3]])
            a_out = torch_resize(a_out)
            I_rec = j_out * t_out + (1 - t_out) * a_out

            if not os.path.exists(opt.output_folder):
                os.makedirs(opt.output_folder)
                # os.mkdir(opt.output_folder + 'I/')
                # os.mkdir(opt.output_folder + 'J/')
                # os.mkdir(opt.output_folder + 'T/')
                # os.mkdir(opt.output_folder + 'A/')
            # i_out_np = np.clip(torch_to_np(I_rec), 0, 1)
            j_out_np = np.clip(torch_to_np(j_out), 0, 1)
            # t_out_np = np.clip(torch_to_np(t_out), 0, 1)
            # a_out_np = np.clip(torch_to_np(a_out), 0, 1)
            # my_save_image(name[0], i_out_np, opt.output_folder + 'I/')
            # my_save_image(name[0], j_out_np, opt.output_folder + 'J/')
            my_save_image(name[0], j_out_np, opt.output_folder)
            # my_save_image(name[0], t_out_np, opt.output_folder + 'T/')
            # my_save_image(name[0], a_out_np, opt.output_folder + 'A/')
    if (len(times) > 1):
        print("\nTotal samples: %d" % len(testing_data_loader))
        # accumulate frame processing times (without bootstrap)
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
        print("Time taken: %.3f sec at %0.3f fps" % (Ttime, 1. / Mtime))

eval()
