import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image


def download_image_as_bytes(url, max_file_size=0):
    """根据 URL 下载图片。

    Args:
        url: string, 图片URL.
        max_file_size: integer, 图片文件最大值，0则不限制图片大小。

    Returns:
        bytes, 图片

    Raises:
        超过最大图片文件大小限制
        ImageUrlDownloadFailError 图片通过 URL 下载失败
    """
    try:
        # requests, streaming request and streaming uploads
        # https://requests.readthedocs.io/en/stable/user/advanced/
        res = requests.get(url, allow_redirects=True, stream=True)
        if res.status_code != 200:
            raise Exception('ImageUrlDownloadFailError')
    except:
        raise Exception('ImageUrlDownloadFailError')

    file_size = int(res.headers['content-length'])
    if 0 < max_file_size < file_size:
        raise Exception(
            f'image size must less than {max_file_size / 1024 / 1024}MB, '
            f'but get {file_size / 1024 / 1024}MB, url is {url}'
        )

    image = res.content
    res.close()
    return image


def get_request_images_as_file(
        image_files=None, image_base64s=None, image_urls=None, max_file_size=0
):
    """由于允许客户端以 file, base64, url 三种方式发送图片，但同一请求只能以其中一种方式传输图片，
    判断上传图片方式，并将数据转换成 bytes 类型的图片文件。

    Args:
        image_files: list of object which implements `.read()`, `.tell()`, `.seek()`, such as
            werkzeug.FileStorage, 图片文件
        image_base64s: list of string, base64 编码后的图片文件
        image_urls: list of string, 图片URL地址
        max_file_size: integer, 单张图片文件大小最大值，如果超过了会抛 InvalidContentLengthError 异常

    Returns:
        Tuple[List[bytes], string], 图片文件对象，传入的图片数据类型，包括 "file", "base64", "url".

    Raises:
        InvalidContentLengthError, 超过最大图片文件大小限制
        InvalidArgumentError, 图片参数或值为空
        ImageUrlDownloadFailError, 图片通过 URL 下载失败
    """
    if image_files and len(image_files) > 0 and None not in image_files:
        image_bytes = [f.read() for f in image_files]
        type_ = 'file'

    elif image_base64s and len(image_base64s) > 0 and None not in image_base64s:
        image_bytes = [base64.b64decode(s) for s in image_base64s]
        type_ = 'base64'

    elif image_urls and len(image_urls) > 0 and None not in image_urls:
        image_bytes = [download_image_as_bytes(url, max_file_size=max_file_size)
                       for url in image_urls]
        type_ = 'url'

    else:
        raise Exception('no image')

    if max_file_size > 0:
        for idx, f in enumerate(image_bytes):
            if len(f) > max_file_size:
                raise Exception(f'image size must less than {max_file_size / 1024 / 1024}MB, '
                                f'but image-{idx + 1} is {len(f) / 1024 / 1024}MB.')
    return image_bytes, type_


def mnist_preprocess(images):
    results = []
    for img in images:
        if isinstance(img, bytes):
            image_file = BytesIO(img)
        else:
            image_file = img

        image = Image.open(image_file)
        image = image.resize((28, 28))
        image = image.convert('L')
        image = np.asarray(image)[:, :] / 255.
        results.append(image)
    return np.asarray(results)
