#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cmath>
#include <array>
#include <list>

using namespace cv;

// A general function that applies an x and y filter and xy filter
//, then displays all 3 results
void applyAndDisplay(Mat image, Mat xFilter, Mat yFilter, std::string name) {
  // Displaying the original image
  namedWindow("Display Image", WINDOW_NORMAL );
  imshow("Display Image", image);
  waitKey(0);

  // New mat to hold our gray
  Mat img_gray;

  cvtColor(image, img_gray, COLOR_BGR2GRAY);

  // Blurring provides better edge detection, so apply it
  Mat img_blur;
  GaussianBlur(img_gray, img_blur, Size(3, 3), 0);

  Mat imgX;
  Mat imgY;
  Mat imgXY;
  filter2D(img_blur, imgX, -1, xFilter, Point(-1, -1), 0, 4);
  filter2D(img_blur, imgY, -1, yFilter, Point(-1, -1), 0, 4);
  // Applying y to the already x-filtered image to get the combination image
  filter2D(imgX, imgXY, -1, yFilter, Point(-1, -1), 0, 4);

  imshow(name + " X", imgX);
  waitKey(0);
  imshow(name + " Y", imgY);
  waitKey(0);
  imshow(name + " XY", imgXY);
  waitKey(0);
}

void applyPrewitt(Mat image) {
  // Prewitt edge detection
  Mat prewX = (Mat_<double>(3,3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
  Mat prewY = (Mat_<double>(3,3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);

  applyAndDisplay(image, prewX, prewY, "Prewitt");
}

void applySobel(Mat image) {
  // Sobel edge detection
  Mat sobelX = (Mat_<double>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
  Mat sobelY = (Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

  applyAndDisplay(image, sobelX, sobelY, "Sobel");
}

// Laplacian is a bit special, since it actually only uses one filter
// for all cases. So, instead of x and y, two different filter types are
// used
void applyLaplacian(Mat image) {
  // Laplacian edge detection
  Mat lapOne = (Mat_<double>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
  Mat lapTwo = (Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);

  applyAndDisplay(image, lapOne, lapTwo, "Laplacian");
}

int getIntensity(Mat img, int col, int row) {
  return (int)img.at<uchar>(row, col);
}

// Given a BGR image, extracts FAST edges and draws a circle on
// each one of those points
void getEdgePoints(Mat image) {
  // Making a copy to draw points on it
  Mat copy = image.clone();

  // Converting to gray scale
  Mat img_gray;
  cvtColor(image, img_gray, COLOR_BGR2GRAY, 1);

  // Blurring provides better edge detection, so apply it
  Mat img_blur;
  GaussianBlur(img_gray, img_blur, Size(3, 3), 0);

  // Defining our bresenham circle as having
  // radius 3. This is just the way this
  // detector is configured. Note, this implementation
  // of pixel selection only works for radius 3; hasn't
  // been generalized just yet
  int bresRadius = 3;

  // The number of points that need to be greater or lower.
  // This is arbitrary, we can learn it later
  int n = 12;

  // This is the threshold value. Arbitrarily selected
  int thresh = 40;

  // This list stores all corner cooridnates we've found
  std::list<std::array<int, 2>> corners = std::list<std::array<int, 2>>();

  // A brehman with rad 3 has 16 points we look at. We're declaring it out here
  // to minimize declaration calls

  std::array<int, 16> intensities = std::array<int, 16>();
  // Iterating column first:
  for(int c = bresRadius; c < img_blur.cols - bresRadius; c++) {
      // Iterating row second
      for(int r = bresRadius; r < img_blur.rows - bresRadius; r++) {
        // Getting all the intensities into an array by hardcoding
        intensities[0] = getIntensity(img_blur, c, r - 3);
        intensities[1] = getIntensity(img_blur, c + 1, r - 3);
        intensities[2] = getIntensity(img_blur, c + 2, r - 2);
        intensities[3] = getIntensity(img_blur, c + 3, r - 1);
        intensities[4] = getIntensity(img_blur, c + 3, r);
        intensities[5] = getIntensity(img_blur, c + 3, r + 1);
        intensities[6] = getIntensity(img_blur, c + 2, r + 2);
        intensities[7] = getIntensity(img_blur, c + 1, r + 3);
        intensities[8] = getIntensity(img_blur, c, r + 3);
        intensities[9] = getIntensity(img_blur, c - 1, r + 3);
        intensities[10] = getIntensity(img_blur, c - 2, r + 2);
        intensities[11] = getIntensity(img_blur, c - 3, r + 1);
        intensities[12] = getIntensity(img_blur, c - 3, r);
        intensities[13] = getIntensity(img_blur, c - 3, r - 1);
        intensities[14] = getIntensity(img_blur, c - 2, r - 2);
        intensities[15] = getIntensity(img_blur, c - 1, r - 3);

        // Now that we have all intensities, calculate how many are greater or lower than the center point
        int cIntensity = getIntensity(img_blur, c, r);

        // These counts count up how many contiguous intensity values
        // were higher or lower than the threshold.
        int lowCount = 0;
        int highCount = 0;

        for(int i = 0; i < intensities.size(); i++) {
          int current = intensities[i];

          // Printing each intensity
          // std::cout << current << " ";
          
          // Updating counts based on current intensities
          if(current > cIntensity + thresh) {
            lowCount = 0;
            highCount++;
          } else if(current < cIntensity - thresh) {
            lowCount++;
            highCount = 0;
          } else {
            lowCount = 0;
            highCount = 0;
          }

          // Now that updates are done, if this satisfies a corner's requirements, add to corner list
          if(lowCount >= n || highCount >= n) {
            std::array<int, 2> coords = { c , r };
            
            // Drawing on image
            circle(copy, Point(c, r), 1, Scalar( 0, 0, 255 ), LINE_8);

            // Pushing to corners list
            corners.push_back(coords);

            break;
          }
        }
      }
  }

  // Resizing image to some multiplier of their original size
  const double MULT = .5;

  resize(img_blur, img_blur, Size(), MULT, MULT);
  resize(copy, copy, Size(), MULT, MULT);

  // Displaying the image
  namedWindow("Blur", WINDOW_NORMAL );
  imshow("Blur", img_blur);
  waitKey(0);

  namedWindow("Corners", WINDOW_NORMAL );

  imshow("Corners", copy);
  waitKey(0);
}

int main(int argc, char** argv ) {
  if ( argc != 2 ) {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }
  // Just showing the image
  Mat image;
  image = imread( argv[1], 1 );
  if ( !image.data ) {
      printf("No image data \n");
      return -1;
  }

  getEdgePoints(image);
  // applyPrewitt(image);
  // applySobel(image);
  // applyLaplacian(image);
  
  return 0;
}
