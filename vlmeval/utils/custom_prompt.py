from ..smp import *
from .dataset_config import img_root_map
from abc import abstractmethod


class CustomPrompt:

    @abstractmethod
    def use_custom_prompt(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, line, dataset):
        raise NotImplementedError

    def dump_image(self, line, dataset):
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)
        img_root = osp.join(ROOT, 'images', img_root_map[dataset] if dataset in img_root_map else dataset)
        os.makedirs(img_root, exist_ok=True)
        if isinstance(line['image'], list):
            tgt_path = []
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = osp.join(img_root, im_name)
                if not read_ok(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = osp.join(img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
        return tgt_path

    def sample_image_from_video(self, video, dataset, sample_frame_num=8):
        video_path = osp.join(self.data_path, f'video/{video}.mp4')
        if osp.exists(video_path):
            vid = decord.VideoReader(video_path)
            step_size = len(vid) / (sample_frame_num + 1)
            indices = [int(i * step_size) for i in range(1, sample_frame_num + 1)]
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]

            img_root = osp.join(self.data_path, 'images', video)
            os.makedirs(img_root, exist_ok=True)
            img_paths = [osp.join(img_root, f'{i}_{sample_frame_num}.jpg') for i in range(1, sample_frame_num + 1)]
            for im, pth in zip(images, img_paths):
                im.save(pth)

            return img_paths
        else:
            raise FileNotFoundError(f"Video {video} not found")



