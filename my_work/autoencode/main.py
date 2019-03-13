from scipy.io import loadmat
from collections import Counter
import pandas as pd
from autoencoder.model import *
from autoencoder.data_process import *
from autoencoder.discriminant_method import io_l2_error


def main():
    img = loadmat('../data2/Salinas_corrected.mat')['salinas_corrected']
    gt = loadmat('../data2/Salinas_gt.mat')['salinas_gt']
    n_bands = img.shape[-1]
    class_count = Counter(gt.reshape((gt.shape[0] * gt.shape[1],)))
    print(class_count)
    print(len(class_count))

    for c in range(1, len(class_count)):
        print('select {}th class as background'.format(c))
        background = c
        model = AutoEncoder(n_bands)
        train_data, test_data_0 = select_background(img, gt, selected_background=background)
        print(train_data.shape, len(test_data_0))
        train_data, train_band_min, train_band_max = normalized_train(train_data)
        model, train_decoded, loss_log = train_autoencoder(model, train_data, 0.00001, 5000, use_cuda=True)
        model.cpu()
        model.eval()

        res = []
        for i in range(1, len(class_count)):
            if i == background:
                continue
            test_data, test_label = get_testset(img, gt, test_data_0, i)
            test_data = normalized_test(test_data, train_band_min, train_band_max)
            test_x = Variable(torch.FloatTensor(test_data))
            test_decoded = model(test_x)
            test_decoded = test_decoded.data.numpy()
            l2_res = io_l2_error(test_decoded, test_data)
            res_metrics = pre_metrics(np.array(test_label), l2_res, train_decoded.cpu().data.numpy(), train_data,
                                      threshold_rate=0.9)
            cur = [i, res_metrics['accuracy'], res_metrics['False alarm rate'], res_metrics['Recognition rate']]
            res.append(cur)
        res = pd.DataFrame(res, columns=['class', 'accuracy', 'False alarm rate', 'Recognition rate'])
        save_name = './result/salinas/background_{}.csv'.format(c)
        res.to_csv(save_name, index=False)


if __name__ == '__main__':
    main()

