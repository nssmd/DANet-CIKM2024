from data_loader.data_loader import Dataset_parking_data_h,Dataset_parking_data_h_at
from torch.utils.data import DataLoader

data_dict = {
    "parking_h":Dataset_parking_data_h,
    "parking_h_at":Dataset_parking_data_h_at
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    pred_len=args.pred_len
    if flag == 'train' and args.isRNN == 1:
        pred_len = 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only,
        begin_index=args.begin_index,
        end_index=args.end_index,
        test_index=args.test_index,
        random_size = args.random_size
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
