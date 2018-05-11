from __future__ import print_function
from itertools import product
from concurrent import futures
import numpy as np

from .file import File
from ._z5py import read_chunk, write_chunk


# ND blocking generator
def blocking(shape, block_shape):
    if len(shape) != len(block_shape):
        raise RuntimeError("Invalid number of dimensions.")
    ranges = [range(sha // bsha if sha % bsha == 0 else sha // bsha + 1)
              for sha, bsha in zip(shape, block_shape)]
    start_points = product(*ranges)
    for start_point in start_points:
        positions = [sp * bshape for sp, bshape in zip(start_point, block_shape)]
        yield tuple(slice(pos, min(pos + bsha, sha))
                    for pos, bsha, sha in zip(positions, block_shape, shape))


# read data from the bounding-box with multiple threads
def read_multithreaded(dataset, bounding_box, n_threads):
    assert isinstance(bounding_box, tuple)
    assert all(isinstance(slice, bb) for bb in bounding_box)
    chunks = dataset.chunks

    # allocate the output data
    offset = tuple(bb.start for bb in bounding_box)
    out_shape = tuple(bb.stop - bb.start for bb in bounding_box)
    ndim = len(out_shape)
    out = np.zeros(out_shape, dtype=dataset.dtype)

    # find all the chunk ids that have overlap with the bounding box
    chunk_ids = ds.get_chunk_requests(offset, out_shape)

    def read_single_chunk(chunk_id):
        # directly request the chunk
        chunk_data = read_chunk(dataset, chunk_id)
        data_offset = tuple(c_id * cshape - off
                            for c_id, cshape, off in zip(chunk_id, chunks, offset))
        data_bb = tuple(slice(doff, doff + sha) for doff, sha in zip(data_offset, chunk_data.shape))

        # clip data and data bounding box if chunk is not fully contained
        # on the left
        clip_left = tuple(db.start < 0 for db in data_bb)
        if any(clip_left):
            for dim, clip in enumerate(clip_left):
                if clip:
                    overhang = offset[dim] - chunk_id[dim] * chunks[dim]
                    clip_bb = tuple(slice(None) if ii != dim else slice(overhang, chunk_data.shape[dim])
                                    for ii in range(ndim))
                    chunk_data = chunk_data[clip_bb]
                    data_bb[dim].start = 0

        # clip data and data bounding box if chunk is not fully contained
        # on the right
        clip_right = tuple(db.stop > dshape for db, dshape in (data_bb, out_shape))
        if any(clip_right):
            for dim, clip in enumerate(clip_right):
                if clip:
                    overhang = chunk_id[dim] * chunks[dim] - offset[dim]
                    clip_bb = tuple(slice(None) if ii != dim else slice(0, chunk_data.shape[dim] - overhang)
                                    for ii in range(ndim))
                    chunk_data = chunk_data[clip_bb]
                    data_bb[dim].stop = out_shape[dim]

        out[data_bb] = chunk_data

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(read_single_chunk, chunk_id) for chunk_id in chunk_ids]
        [t.result() for t in tasks]

    return out


# write input_ multithreaded, beginning from offset
def write_multithreaded(dataset, input_, offset, n_threads):
    pass


# TODO zarr support
# rechunk a n5 dataset
# also supports new compression opts
def rechunk(in_path,
            out_path,
            in_path_in_file,
            out_path_in_file,
            out_chunks,
            n_threads,
            out_blocks=None,
            dtype=None,
            **new_compression):
    f_in = File(in_path, use_zarr_format=False)
    f_out = File(out_path, use_zarr_format=False)

    # if we don't have out-blocks explitictly given,
    # we iterate over the out chunks
    if out_blocks is None:
        out_blocks = out_chunks

    ds_in = f_in[in_path_in_file]
    # if no out dtype was specified, use the original dtype
    if dtype is None:
        dtype = ds_in.dtype

    shape = ds_in.shape
    compression_opts = ds_in.compression_options
    compression_opts.update(new_compression)
    ds_out = f_out.create_dataset(out_path_in_file,
                                  dtype=dtype,
                                  shape=shape,
                                  chunks=out_chunks,
                                  **compression_opts)

    def write_single_chunk(roi):
        ds_out[roi] = ds_in[roi].astype(dtype, copy=False)

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(write_single_chunk, roi)
                 for roi in blocking(shape, out_blocks)]
        [t.result() for t in tasks]

    # copy attributes
    in_attrs = ds_in.attrs
    out_attrs = ds_out.attrs
    for key, val in in_attrs.items():
        out_attrs[key] = val
