#include "serial/visualization.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "serial/signal_analysis.hpp"
#include "types.hpp"

// ______________________________________________________________________________________________

void draw_piece_overlays(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<PuzzlePiece>& pieces,
    const std::string& output_path)
{
    // wrapping the raw rgb data in an OpenCV matrix without copying
    // CV_8UC3 = 8 bit unsigned, 3 channels (RGB)
    cv::Mat image(height, width, CV_8UC3,
        const_cast<uint8_t*>(rgb_data));

    // OpenCV works in BGR, so we convert from RGB
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);

    // iterating over all pieces
    for (const PuzzlePiece& piece : pieces)
    {
        const Region& region = piece.region;
        const CoordinateVector<int>& contour = piece.contour;

        // ______________________________________________________________________________________________

        // drawing the contour in red
        // contour points are (x, y) after the coordinate swap in find_contour_chain_approx_simple
        for (size_t i = 0; i < contour.size(); ++i)
        {
            size_t next = (i + 1) % contour.size();
            cv::Point p1(contour[i].a, contour[i].b);
            cv::Point p2(contour[next].a, contour[next].b);
            cv::line(bgr, p1, p2, cv::Scalar(0, 0, 255), 1);
        }

        // ______________________________________________________________________________________________

        // drawing the bounding box in light green
        cv::Rect bbox(region.x, region.y, region.width, region.height);
        cv::rectangle(bgr, bbox, cv::Scalar(144, 238, 144), 2);

        // ______________________________________________________________________________________________

        // drawing the label text in blue
        const std::string piece_label = piece.class_label.empty()
            ? edges_to_string(piece.edge_labels)
            : piece.class_label;
        std::string text = "E" + std::to_string(region.label) + ": " + piece_label;
        cv::Point text_pos(region.x, std::max(0, region.y - 8));
        cv::putText(
            bgr,
            text,
            text_pos,
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 0, 0),
            1,
            cv::LINE_AA
        );
    }

    if (!output_path.empty())
    {
        cv::imwrite(output_path, bgr);
    }
}
