#pragma once
#include "z5/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "xtensor/xeval.hpp"


namespace z5 {
namespace multiarray {

    template<typename T, typename ITER>
    void writeScalar(const Dataset & ds, ITER roiBeginIter, ITER roiShapeIter, const T val) {

        // get the offset and shape of the request and check if it is valid
        types::ShapeType offset(roiBeginIter, roiBeginIter+ds.dimension());
        types::ShapeType shape(roiShapeIter, roiShapeIter+ds.dimension());
        ds.checkRequestShape(offset, shape);
        ds.checkRequestType(typeid(T));

        // get the chunks that are involved in this request
        std::vector<types::ShapeType> chunkRequests;
        ds.getChunkRequests(offset, shape, chunkRequests);

        types::ShapeType offsetInRequest, requestShape, chunkShape, offsetInChunk;
        // out buffer holding data for a single chunk
        types::ShapeType bufferShape;
        // N5-Axis order: we need to reverse the max chunk shape
        if(ds.isZarr()) {
           bufferShape = types::ShapeType(ds.maxChunkShape().begin(), ds.maxChunkShape().end());
        } else {
           bufferShape = types::ShapeType(ds.maxChunkShape().rbegin(), ds.maxChunkShape().rend());
        }
        xt::xarray<T> buffer(bufferShape, val);

        // iterate over the chunks and write the buffer
        for(const auto & chunkId : chunkRequests) {

            std::cout << "Writing chunk " << chunkId << std::endl;

            bool completeOvlp = ds.getCoordinatesInRequest(chunkId, offset, shape, offsetInRequest, requestShape, offsetInChunk);
            ds.getChunkShape(chunkId, chunkShape);
            // N5-Axis order: we need to reverse the chunk shape internally
            if(!ds.isZarr()) {
                std::reverse(chunkShape.begin(), chunkShape.end());
                std::reverse(offsetInChunk.begin(), offsetInChunk.end());
                std::reverse(requestShape.begin(), requestShape.end());
            }

            // reshape buffer if necessary
            if(bufferShape != chunkShape) {
                std::cout << "Reshapeing to " << chunkShape << std::endl;
                buffer.reshape(chunkShape);
                bufferShape = chunkShape;
                buffer = val;
            }

            // request and chunk overlap completely
            // -> we can write the whole chunk
            if(completeOvlp) {
                ds.writeChunk(chunkId, buffer.raw_data());
            }

            // request and chunk overlap only partially
            // -> we can only write partial data and need
            // to preserve the data that will not be written
            else {
                // load the current data into the buffer
                ds.readChunk(chunkId, buffer.raw_data());
                // overwrite the data that is covered by the view
                xt::slice_vector bufSlice(buffer);
                sliceFromRoi(bufSlice, offsetInChunk, requestShape);
                auto bufView = xt::dynamic_view(buffer, bufSlice);
                bufView = val;
                ds.writeChunk(chunkId, buffer.raw_data());
            }
        }
    }


    // unique ptr API
    template<typename T, typename ITER>
    inline void writeScalar(std::unique_ptr<Dataset> & ds, ITER roiBeginIter, ITER roiShapeIter, const T val) {
       writeScalar(*ds, roiBeginIter, roiShapeIter, val);
    }
}
}
