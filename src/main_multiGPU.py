import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import shutil
import yaml
import torch
import Models
torch.cuda.empty_cache()
from documenter import Documenter
def main():
    parser = argparse.ArgumentParser(description='Fast Calorimeter Simulation with CaloDreamer')
    parser.add_argument('param_file', help='yaml parameters file')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('-d', '--model_dir', default=None)
    parser.add_argument('-ep', '--epoch', default='')
    parser.add_argument('-g', '--generate', action='store_true', default=False)
    parser.add_argument('--which_cuda', default=0)
    parser.add_argument('--sampling_type', default='ddpm')
    args = parser.parse_args()

    args = parser.parse_args()
    print(args.param_file)

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    #use_cuda = torch.cuda.is_available() and args.use_cuda
    

    # Select device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    if args.model_dir:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])

    try:
        shutil.copy(args.param_file, doc.get_file('params.yaml'))
    except shutil.SameFileError:
        pass
 
    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)


    model_name = params.get("model", "TBD")
    print("I am here", model_name)

    try:
        model = getattr(Models, model_name)(params, device, doc)
    except AttributeError:
        raise NotImplementedError(f"build_model: Model class {model_name} not recognised")

    # Move model to GPU
    model = model.to(device)

    # Wrap with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        # print(f"Using {torch.cuda.device_count()} GPUs!")
        # model = torch.nn.DataParallel(model)
        
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank
        )


    # Now you can continue with training or sampling
    if not args.plot:
        model.module.run_training()
    else:
        if args.generate:
            x, c = model.module.sample_trained_model_energy()
        else:
            model.module.plot_samples()
if __name__=='__main__':
    main()
