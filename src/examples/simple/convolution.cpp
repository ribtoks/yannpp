#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/layers/convolutionlayer.h>
#include <yannpp/common/log.h>
#include <yannpp/common/utils.h>

using namespace yannpp;

#define SCALE (1.f)
//#define SCALE (42.333553f)

int fill_array(array3d_t<float> &arr, int start=0) {
    int i = start;
    auto slice = arr.slice();
    auto it = slice.iterator();
    for (; it.is_valid(); ++it, i++) {
        slice.at(*it) = i/SCALE;
    }

    return (i - 1);
}

array3d_t<float> conv2d(std::vector<array3d_t<float>> &&filters, array3d_t<float> input, yannpp::padding_type padding) {
    const auto &filter_shape = filters[0].shape();
    shape3d_t conv_shape(FILTER_DIM(input.shape().x(), filter_shape.x(), 1),
                         FILTER_DIM(input.shape().y(), filter_shape.y(), 1),
                         filters.size());
    bool same = padding == padding_type::same;

    shape3d_t output_shape = same ? shape3d_t(input.shape().x(), input.shape().y(), filters.size()) : conv_shape;
    array3d_t<float> result(output_shape, 0);

    const int pad_x = same ? utils::get_left_padding(input.shape(), filter_shape, 1) : 0;
    const int pad_y = same ? utils::get_top_padding(input.shape(), filter_shape, 1) : 0;

    const int fsize = filters.size();
    // perform convolution for each filter
    for (int fi = 0; fi < fsize; fi++) {
        auto filter = filters[fi].slice();
        // 2D loop over the input and calculation convolution of input and current filter
        // convolution is S(i, j) = (I ∗ K)(i, j) = Sum[ I(m, n)K(i − m, j − n) ]
        // which is commutative i.e. (I ∗ K)(i, j) = Sum[ I(i - m, j - n)K(m, n) ]
        // where I is input and K is kernel (filter weights)
        for (int y = 0; y < output_shape.y(); y++) {
            int ys = y * 1 - pad_y;

            for (int x = 0; x < output_shape.x(); x++) {
                int xs = x * 1 - pad_x;
                // in this case cross-correlation (I(m, n)K(i + m, j + n)) is used
                // (kernel is not rot180() flipped for the convolution, not commutative)
                // previous formula (w*x + b) is used with convolution instead of product
                result(x, y, fi) =
                        dot<float>(
                            input.slice(
                                index3d_t(xs, ys, 0),
                                index3d_t(xs + filter_shape.x() - 1,
                                          ys + filter_shape.y() - 1,
                                          input.shape().z() - 1)),
                            filter);
            }
        }
    }

    return result;
}

array3d_t<float> create_input(shape3d_t const &shape) {
    array3d_t<float> arr(shape, 0.f);
    fill_array(arr);
    return arr;
}

std::vector<array3d_t<float>> create_filters(int count, shape3d_t const &shape) {
    std::vector<array3d_t<float>> filters;

    for (auto fi = 0; fi < count; fi++) {
        filters.emplace_back(shape, 0.f);
    }

    int index = 0;

    for (auto height = 0; height < shape.y(); height++) {
        for (auto width = 0; width < shape.x(); width++) {
            for (auto depth = 0; depth < shape.z(); depth++) {
                for (auto fi = 0; fi < count; fi++, index++) {
                    filters[fi](height, width, depth) = index/SCALE;
                }
            }
        }
    }

    return filters;
}

int main(int argc, char* argv[]) {
    using namespace yannpp;

    array3d_t<float> rgb(shape3d_t(5, 5, 3), 0.f);
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            for (int z = 0; z < 3; z++) {
                rgb(x, y, z) = z;
            }
        }
    }

    for (int i = 0; i < 5*5*3; i++) {
        std::cout << rgb.data()[i] << " ";
    }
    std::cout << std::endl;

    log(rgb);
    //return 0;

    log("-----------------------------");

    // 29730.
    log(conv2d(create_filters(1, shape3d_t(3,3,5)),
               create_input(shape3d_t(3,3,5)),
               padding_type::valid));

    log("-----------------------------");

//    [[[[ 8970.]
//       [15080.]
//       [10470.]]

//      [[19080.]
//       [29370.]
//       [19080.]]

//      [[10470.]
//       [15080.]
//       [ 8970.]]]]
    log(conv2d(create_filters(1, shape3d_t(3,3,5)),
               create_input(shape3d_t(3,3,5)),
               padding_type::same));

    log("-----------------------------");

//    [[[[ 43770.]
//       [ 48720.]
//       [ 53670.]]

//      [[ 68520.]
//       [ 73470.]
//       [ 78420.]]

//      [[ 93270.]
//       [ 98220.]
//       [103170.]]]]
    log(conv2d(create_filters(1, shape3d_t(3,3,5)),
               create_input(shape3d_t(5,5,5)),
               padding_type::valid));

    log("-----------------------------");

//    [[[[306390. 307830. 309270. 310710. 312150. 313590. 315030.]
//       [341040. 342705. 344370. 346035. 347700. 349365. 351030.]
//       [375690. 377580. 379470. 381360. 383250. 385140. 387030.]]

//      [[479640. 482205. 484770. 487335. 489900. 492465. 495030.]
//       [514290. 517080. 519870. 522660. 525450. 528240. 531030.]
//       [548940. 551955. 554970. 557985. 561000. 564015. 567030.]]

//      [[652890. 656580. 660270. 663960. 667650. 671340. 675030.]
//       [687540. 691455. 695370. 699285. 703200. 707115. 711030.]
//       [722190. 726330. 730470. 734610. 738750. 742890. 747030.]]]]
    log(conv2d(create_filters(7, shape3d_t(3,3,5)),
               create_input(shape3d_t(5,5,5)),
               padding_type::valid));

    log("-----------------------------");

//    [[[[ 90440.  90780.  91120.  91460.  91800.  92140.  92480.]
//       [144410. 144995. 145580. 146165. 146750. 147335. 147920.]
//       [175385. 176120. 176855. 177590. 178325. 179060. 179795.]
//       [206360. 207245. 208130. 209015. 209900. 210785. 211670.]
//       [135240. 135880. 136520. 137160. 137800. 138440. 139080.]]

//      [[206010. 206895. 207780. 208665. 209550. 210435. 211320.]
//       [306390. 307830. 309270. 310710. 312150. 313590. 315030.]
//       [341040. 342705. 344370. 346035. 347700. 349365. 351030.]
//       [375690. 377580. 379470. 381360. 383250. 385140. 387030.]
//       [236460. 237795. 239130. 240465. 241800. 243135. 244470.]]

//      [[334635. 336270. 337905. 339540. 341175. 342810. 344445.]
//       [479640. 482205. 484770. 487335. 489900. 492465. 495030.]
//       [514290. 517080. 519870. 522660. 525450. 528240. 531030.]
//       [548940. 551955. 554970. 557985. 561000. 564015. 567030.]
//       [338835. 340920. 343005. 345090. 347175. 349260. 351345.]]

//      [[463260. 465645. 468030. 470415. 472800. 475185. 477570.]
//       [652890. 656580. 660270. 663960. 667650. 671340. 675030.]
//       [687540. 691455. 695370. 699285. 703200. 707115. 711030.]
//       [722190. 726330. 730470. 734610. 738750. 742890. 747030.]
//       [441210. 444045. 446880. 449715. 452550. 455385. 458220.]]

//      [[233240. 235080. 236920. 238760. 240600. 242440. 244280.]
//       [311360. 314195. 317030. 319865. 322700. 325535. 328370.]
//       [326585. 329570. 332555. 335540. 338525. 341510. 344495.]
//       [341810. 344945. 348080. 351215. 354350. 357485. 360620.]
//       [194040. 196180. 198320. 200460. 202600. 204740. 206880.]]]]
    log(conv2d(create_filters(7, shape3d_t(3,3,5)),
               create_input(shape3d_t(5,5,5)),
               padding_type::same));

    return 0;
}
