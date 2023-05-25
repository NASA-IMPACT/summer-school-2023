import json
import os
import datetime
import string
import pandas as pd
import boto3
from botocore.client import Config
from tqdm.notebook import tqdm, trange
import ipywidgets as widgets
import subprocess
import glob
from pprint import pprint


def get_cluster_details(path_to_shared_volume='/opt/app-root/src/data/'):
    narg = json.loads(os.environ.get('NOTEBOOK_ARGS').split('--ServerApp.tornado_settings=')[-1])
    login_url = narg['hub_host'].replace('rhods-dashboard-redhat-ods-applications','oauth-openshift') + '/oauth/token/request'
    namespace = narg['hub_prefix'].replace('/projects/','')
    return login_url, namespace, path_to_shared_volume

def generate_config(project_name, conf, template_file):
    
    experiment_name = project_name + '-exp-' + datetime.datetime.now().strftime('%Y%m%d-%H%M')
    conf['exp_name'] = experiment_name

    experiment_filepath = '/opt/app-root/src/data/' + project_name + '/configs/' + experiment_name + '_config.py'

    conf['iter_per_eval'] = str(5*int(conf['number_training_files']/int(conf['batch_size'])))
    conf['num_iterations'] = str(conf['num_epochs']*int(conf['number_training_files']/int(conf['batch_size'])))
    
    print(conf)
    
    with open(template_file) as t:
        template = string.Template(t.read())

    final_output = template.substitute(**conf)

    with open(experiment_filepath, "w") as output:
        output.write(final_output)
    
    return experiment_name, experiment_filepath



def create_project_folders(project_name, path_to_shared_volume="/opt/app-root/src/data/"):
    try:
        # os.system("./create_project_folders.sh " + project_name + " " + path_to_shared_volume)
        
        os.mkdir(path_to_shared_volume + project_name) 
        print("Created overall project folder at: " + path_to_shared_volume + project_name)
        
        folders = ['configs', 'fine-tune-checkpoints', 'gfm-models', 'inference', 'training-data']
        
        for f in folders:
            os.mkdir(path_to_shared_volume + project_name + '/' + f) 
            print("Created " + f + " folder at: " + path_to_shared_volume + project_name)    
        
    except:
        print("failed")
        

def create_s3_client(access_key, access_secret, endpoint_url="https://s3.us-south.cloud-object-storage.appdomain.cloud"):        
    s3 = boto3.client(
        "s3",
        endpoint_url = endpoint_url,
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_ACCESS_KEY_SECRET"),
        config = Config(signature_version="s3v4"),
        region_name = "us-south",
    )        
    return s3
        
def download_s3_dir(prefix, local, bucket, client, number_of_files=None, random_state=17):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
            
    # Add random sampling of files here - use pandas
    if number_of_files is not None:
        df = pd.DataFrame(keys, columns=['filename'])
        df['type'] = ['mask' if 'mask' in X else "merged" for X in keys]
        fdf = df[df.type=="merged"].reset_index()
        sfdf = fdf.sample(n=number_of_files, replace=False, weights=None, random_state=17, axis=None, ignore_index=False)

        keys_merged = list(sfdf['filename'].values)
        keys_mask = [X.replace('_merged','.mask') for X in keys_merged]
        selected_keys = keys_merged + keys_mask 
    else:
        selected_keys = keys
            
    for k in trange(len(selected_keys)):
        dest_pathname = os.path.join(local, keys[k])
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, keys[k], dest_pathname)
        

def view_config(experiment_filepath):
    with open(experiment_filepath) as f:
        config_string = f.read()


    return widgets.Textarea(
        value=config_string,
        placeholder='Type something',
        description='Config:',
        rows=20,
        layout=widgets.Layout(width='90%', height='400px'),
        disabled=False
    )

def submit_tune(project_name,
                namespace,
                experiment_name,
                image='quay.io/bedwards-ibm/mmsegmentation-geo:latest',
                num_gpus=1,
                memory_mb=26000):
    
    train_cmd = '''torchx run --workspace "" --scheduler kubernetes_mcad \
               --scheduler_args namespace=''' + namespace + ''',image_repo=''' + image.split(':')[0] + ''' \
               dist.ddp -j 1x1 \
               --image ''' + image + ''' \
               --gpu ''' + str(num_gpus) + ''' --memMB ''' + str(memory_mb) + ''' \
               --mount type=volume,src=data,dst="/data" \
               --script mmsegmentation/tools/train.py -- /data/''' + project_name + '''/configs/''' + experiment_name + '''_config.py \
               --launcher 'pytorch' \
               --cfg-options 'find_unused_parameters'=True --no-validate '''

    print('----------------------------------------------')
    print('Running:')
    print(train_cmd.replace('--','\n     --'))
    print('----------------------------------------------')

    mcad_id = subprocess.check_output(train_cmd, shell=True)
    mcad_id = mcad_id.decode("utf-8").replace("\n","")
    
    return mcad_id

def load_tune_metrics(project_name,
                      experiment_name,
                      path_to_shared_volume='/opt/app-root/src/data/'):
    log_name = glob.glob(path_to_shared_volume + project_name + '/fine-tune-checkpoints/' + experiment_name + '/*.log.json')

    with open(log_name[0]) as fp:
        lines = [line.rstrip('\n') for line in fp]
    metrics = [json.loads(X) for X in lines[1:]]

    train_df = pd.DataFrame.from_records([d for d in metrics if d['mode']=='train'])
    val_df = pd.DataFrame.from_records([d for d in metrics if d['mode']=='val'])
    return train_df, val_df

def submit_test(project_name,
                namespace,
                experiment_name,
                checkpoint='latest.pth',
                image='quay.io/bedwards-ibm/mmsegmentation-geo:latest',
                num_gpus=1,
                memory_mb=8000):
    
    test_cmd = '''torchx run --workspace "" --scheduler kubernetes_mcad \
           --scheduler_args namespace=''' + namespace + ''',image_repo=''' + image.split(':')[0] + ''' \
           dist.ddp -j 1x1 \
           --image ''' + image + ''' \
           --gpu ''' + str(num_gpus) + ''' --memMB ''' + str(memory_mb) + ''' \
           --mount type=volume,src=data,dst="/data" \
           --script mmsegmentation/tools/test.py -- /data/''' + project_name  + '''/fine-tune-checkpoints/''' + experiment_name + '''/''' + experiment_name + '''_config.py \
       "/data/''' + project_name  + '''/fine-tune-checkpoints/''' + experiment_name + '''/''' + checkpoint + '''" --eval "mIoU" --work-dir "/data/''' + project_name  + '''/fine-tune-checkpoints/''' + experiment_name + '''"'''

    print('----------------------------------------------')
    print('Running:')
    print(test_cmd.replace('--','\n     --'))
    print('----------------------------------------------')
    
    test_mcad_id = subprocess.check_output(test_cmd, shell=True)
    test_mcad_id = test_mcad_id.decode("utf-8").replace("\n","")
    
    return test_mcad_id

def get_test_metrics(project_name,
                     experiment_name):
    metric_files = sorted(glob.glob('/opt/app-root/src/data/' + project_name  + '/fine-tune-checkpoints/' + experiment_name + '/eval_single_scale*.json'))

    with open(metric_files[-1]) as fp:
        metrics = json.load(fp)

    pprint(metrics)
    
    return metrics

def submit_inference(project_name,
                namespace,
                experiment_name,
                checkpoint='latest.pth',
                image='quay.io/bedwards-ibm/mmsegmentation-geo:latest',
                num_gpus=1,
                memory_mb=8000,
                bands="[2,1,0,3]"):

    infer_cmd = '''torchx run --workspace "" --scheduler kubernetes_mcad \
           --scheduler_args namespace=''' + namespace + ''',image_repo=''' + image.split(':')[0] + ''' \
           dist.ddp -j 1x1 \
           --image ''' + image + ''' \
           --gpu ''' + str(num_gpus) + ''' --memMB ''' + str(memory_mb) + ''' \
           --mount type=volume,src=data,dst="/data" \
           --script mmsegmentation/tools/geospatial_batch_inference.py -- \
           -config /data/''' + project_name  + '''/fine-tune-checkpoints/''' + experiment_name  + '''/''' + experiment_name + '''_config.py \
           -ckpt "/data/''' + project_name  + '''/fine-tune-checkpoints/''' + experiment_name  + '''/''' + checkpoint + '''" \
           -input "/data/''' + project_name  + '''/inference/" \
           -output "/data/''' + project_name  + '''/inference/pred/''' + experiment_name  + '''"/ \
           -bands "''' + bands + '''" '''
    
    print('----------------------------------------------')
    print('Running:')
    print(infer_cmd.replace('--','\n     --'))
    print('----------------------------------------------')
    
    infer_mcad_id = subprocess.check_output(infer_cmd, shell=True)
    infer_mcad_id = infer_mcad_id.decode("utf-8").replace("\n","")
    
    return infer_mcad_id