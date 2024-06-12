import torch
import torch.distributed as dist
import datetime
from vlmeval.config import supported_VLM, api_models
from vlmeval.utils import TSVDataset, track_progress_rich, split_MMMU
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(work_dir, model_name, dataset_name, index_set=None, api_nproc=4, ignore_failed=False, pack=False, video_sample_frame_num=8):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset = TSVDataset(dataset_name, pack=pack)
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)

    # lt, indices = len(dataset), list(data['index'])
    lt, indices = dataset.packed_length(), list(data['index'])
    if listinstr(['MMBench-Video'], dataset_name):
        structs = [dataset.build_prompt_for_video(i, video_sample_frame_num, api=True) for i in range(lt)]
        indices = dataset.videos if pack else list(data['index'])
        print(f'prompt for {dataset_name} is ready. {lt} prompts in total. {len(indices)} indices in total.')
    else:
        structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    # Corner Case
    if listinstr(['MMMU'], dataset_name):
        structs = [split_MMMU(s) for s in structs]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    # For now, we do not use split_MMMU for MMMU dataset
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    org_res = load(out_file)
    if listinstr(['MMBench-Video'], dataset_name) and pack:
        res = {}
        for k, v in org_res.items():
            try:
                v = v.replace('```json','').replace('```','')
                answer_for_video = eval(v)
                idx_list = data[data['video'] == k]['index'].tolist()
                for id, (idx, ans) in enumerate(answer_for_video.items()):
                    try:
                        new_idx = int(idx)
                    except:
                        print(f'idx {idx} not int')
                        new_idx = idx_list[id]

                    if new_idx in idx_list:
                        res[new_idx] = ans
                    else:
                        print(f'idx {new_idx} not in {idx_list}')
                        res[idx_list[id]] = ans
                
            except Exception as e:
                print(f'error:{e}, parse fail')
                failed_idx_list = data[data['video'] == k]['index'].tolist()
                res.update({idx: FAIL_MSG for idx in failed_idx_list})
        
        res = {int(k): v for k, v in res.items()}
            
    elif index_set is not None:
        res = {k: v for k, v in org_res.items() if k in index_set}
    else:
        res = org_res
    # os.remove(out_file)
    return res


def infer_data(model_name, work_dir, dataset_name, out_file, verbose=False, api_nproc=4, pack=True, video_sample_frame_num=8):
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        dataset = TSVDataset(dataset_name, pack=pack)
    if world_size > 1:
        dist.barrier()
    dataset = TSVDataset(dataset_name, pack=pack)

    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
            break
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            index_set=set(indices),
            api_nproc=api_nproc,
            pack=pack,
            video_sample_frame_num=video_sample_frame_num)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model_name

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # Corner Case
        if listinstr(['MMMU'], dataset_name):
            struct = split_MMMU(struct)

        # For now, we do not use split_MMMU for MMMU dataset
        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset_name, verbose=False, api_nproc=4, ignore_failed=False, pack=True, video_sample_frame_num=8):
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model, work_dir=work_dir, dataset_name=dataset_name, out_file=out_file, verbose=verbose, api_nproc=api_nproc, pack=pack, video_sample_frame_num=video_sample_frame_num)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = TSVDataset(dataset_name, pack=pack).data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
