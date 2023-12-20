
from datasets import load_dataset
from image_classifier import Image_trainer
import torch
from torch.utils.data import DataLoader
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_img(args,config):
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    train_data = None
    args.number_of_diff_lrs = int(config["DEFAULT"]["num_diff_opt"])
    if config["DEFAULT"]["optim"] == "sgd":
      lr = 1e-1
    else:
      lr = 1e-3
    args.opts = {"lr": lr, "opt": config["DEFAULT"]["optim"]}
    args.ds = dataset
    args.split_by = config["DEFAULT"]["split_by"]
    args.update_rule = config["DEFAULT"]["update_rule"]
    args.model = config["DEFAULT"]["model"]
    args.savepth = config["DEFAULT"]["directory"]
    if not os.path.isdir(args.savepth):
        os.mkdir(args.savepth)
    args.combine = float(config["DEFAULT"]["combine"])
    args.c = float(config["DEFAULT"]["c"])
    args.beta = float(config["DEFAULT"]["beta"])
    args.hidden_dims = int(config["DEFAULT"]["num_hidden_dims"])
    args.gradient_accumulation_steps = int(config["DEFAULT"]["gradient_accumulation_steps"])
    cls = config["DEFAULT"]["cls"]

    print("reading dataset")
    valds_name = "test"
    dataset_name = dataset
    resize = True
    if dataset == "imagenet":
      dataset_name = "imagenet-1k"
      num_classes = 1000
      labelname = "label"
      dataname = "image"
      input_dim = 224*224*3
      valds_name = "validation"
    if dataset == "tiny-imagenet":
      dataset_name = "Maysee/tiny-imagenet"
      num_classes = 200
      labelname = "label"
      dataname = "image"
      input_dim = 64*64*3
      valds_name = "valid"
    if dataset == "cifar100":
      num_classes = 100
      labelname = "fine_label"
      dataname = "img"
      input_dim = 32*32*3
    if dataset == "cifar10":
      num_classes = 10
      labelname = "label"
      dataname = "img"
      input_dim = 32*32*3
    if dataset == "mnist":
      resize = False
      num_classes = 10
      labelname = "label"
      dataname = "image"
      input_dim = 28*28*1
    if dataset == "electric":
      num_classes = 2
      input_dim = 7
      from data  import SimpleDataset_electric
      data = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/electricity.csv", split = "train[10%:90%]")
      train_data = SimpleDataset_electric(data)
      data = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/electricity.csv", split = "train[:10%]+train[90%:]")
      val_data = SimpleDataset_electric(data)
    if dataset == "covertype":
      num_classes = 2
      input_dim = 10
      from data  import SimpleDataset_cover
      data = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/covertype.csv", split = "train[10%:90%]")
      train_data = SimpleDataset_cover(data)
      data = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/covertype.csv", split = "train[:10%]+train[90%:]")
      val_data = SimpleDataset_cover(data)
    if dataset == "pol":
      num_classes = 2
      input_dim = 26
      from data  import SimpleDataset_pol
      data = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/pol.csv", split = "train[10%:90%]")
      train_data = SimpleDataset_pol(data)
      data = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/pol.csv", split = "train[:10%]+train[90%:]")
      val_data = SimpleDataset_pol(data)

    if dataset == "cifar10":
      from data import getCifar10
      train_dataloader,val_dataloader = getCifar10(batch_size,args.gradient_accumulation_steps)
    elif dataset == "cifar100":
      from data import getCifar100
      train_dataloader,val_dataloader = getCifar100(batch_size,args.gradient_accumulation_steps)
    elif dataset == "imagenet":
      from data import getImageNet
      train_dataloader,val_dataloader = getImageNet(batch_size,args.gradient_accumulation_steps)



      # print("train images")
      # show_img(train_dataloader)
      # print("val images")
      # show_img(val_dataloader)
    else:
   #   print("hallo")
      if train_data == None:
        dataset = load_dataset(dataset_name, cache_dir="/media/philipkenneweg/Data/datasets")
        from data import SimpleDataset
        train_data = SimpleDataset(dataset["train"],dataname,labelname, resize = resize )
        val_data = SimpleDataset(dataset[valds_name],dataname,labelname , resize = resize)
      train_dataloader = DataLoader(train_data, batch_size=batch_size*args.gradient_accumulation_steps, shuffle=True)
      val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    print("constructing model")
    img_cls = Image_trainer(num_classes,batch_size,args)

    print("start training")
    img_cls.fit(train_dataloader,max_epochs,val_dataloader)
