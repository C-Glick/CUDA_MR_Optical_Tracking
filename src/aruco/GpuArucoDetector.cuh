//
// Created by colton-glick on 4/28/26.
//


#ifndef CUDA_MR_OPTICAL_TRACKING_GPUARUCODETECTOR_H
#define CUDA_MR_OPTICAL_TRACKING_GPUARUCODETECTOR_H
#include "opencv2/objdetect/aruco_detector.hpp"

#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_board.hpp>

namespace cv {
namespace aruco {


/** @brief The main functionality of ArucoDetector class is detection of markers in an image with detectMarkers() method.
 *
 * After detecting some markers in the image, you can try to find undetected markers from this dictionary with
 * refineDetectedMarkers() method.
 *
 * @see DetectorParameters, RefineParameters
 */
class GpuArucoDetector : public cv::aruco::ArucoDetector{
public:
    /** @brief Basic ArucoDetector constructor
     *
     * @param dictionary indicates the type of markers that will be searched
     * @param detectorParams marker detection parameters
     * @param refineParams marker refine detection parameters
     */
    CV_WRAP GpuArucoDetector(const Dictionary &dictionary = getPredefinedDictionary(cv::aruco::DICT_4X4_50),
                          const DetectorParameters &detectorParams = DetectorParameters(),
                          const RefineParameters& refineParams = RefineParameters());

    /** @brief ArucoDetector constructor for multiple dictionaries
     *
     * @param dictionaries indicates the type of markers that will be searched. Empty dictionaries will throw an error.
     * @param detectorParams marker detection parameters
     * @param refineParams marker refine detection parameters
     */
    CV_WRAP GpuArucoDetector(const std::vector<Dictionary> &dictionaries,
                          const DetectorParameters &detectorParams = DetectorParameters(),
                          const RefineParameters& refineParams = RefineParameters());

    /** @brief Basic marker detection
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     *
     * Performs marker detection in the input image. Only markers included in the first specified dictionary
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     */
    CV_WRAP void detectMarkers(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                               OutputArrayOfArrays rejectedImgPoints = noArray()) const;

    /** @brief Marker detection with confidence computation
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param markersConfidence contains the normalized confidence [0;1] of the markers' detection,
     * defined as 1 minus the normalized uncertainty (percentage of incorrect pixel detections),
     * with 1 describing a pixel perfect detection. The confidence values are of type float
     * (e.g. std::vector<float>)
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     *
     * Performs marker detection in the input image. Only markers included in the first specified dictionary
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     */
    CV_WRAP void detectMarkersWithConfidence(InputArray image, OutputArrayOfArrays corners, OutputArray ids, OutputArray markersConfidence,
                               OutputArrayOfArrays rejectedImgPoints = noArray()) const;

    /** @brief Refine not detected markers based on the already detected and the board layout
     *
     * @param image input image
     * @param board layout of markers in the board.
     * @param detectedCorners vector of already detected marker corners.
     * @param detectedIds vector of already detected marker identifiers.
     * @param rejectedCorners vector of rejected candidates during the marker detection process.
     * @param cameraMatrix optional input 3x3 floating-point camera matrix
     * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
     * @param distCoeffs optional vector of distortion coefficients
     * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
     * @param recoveredIdxs Optional array to returns the indexes of the recovered candidates in the
     * original rejectedCorners array.
     *
     * This function tries to find markers that were not detected in the basic detecMarkers function.
     * First, based on the current detected marker and the board layout, the function interpolates
     * the position of the missing markers. Then it tries to find correspondence between the reprojected
     * markers and the rejected candidates based on the minRepDistance and errorCorrectionRate parameters.
     * If camera parameters and distortion coefficients are provided, missing markers are reprojected
     * using projectPoint function. If not, missing marker projections are interpolated using global
     * homography, and all the marker corners in the board must have the same Z coordinate.
     * @note This function assumes that the board only contains markers from one dictionary, so only the
     * first configured dictionary is used. It has to match the dictionary of the board to work properly.
     */
    CV_WRAP void refineDetectedMarkers(InputArray image, const Board &board,
                                       InputOutputArrayOfArrays detectedCorners,
                                       InputOutputArray detectedIds, InputOutputArrayOfArrays rejectedCorners,
                                       InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
                                       OutputArray recoveredIdxs = noArray()) const;

    /** @brief Basic marker detection
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     * @param dictIndices vector of dictionary indices for each detected marker. Use getDictionaries() to get the
     * list of corresponding dictionaries.
     *
     * Performs marker detection in the input image. Only markers included in the specific dictionaries
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     */
    CV_WRAP void detectMarkersMultiDict(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                               OutputArrayOfArrays rejectedImgPoints = noArray(), OutputArray dictIndices = noArray()) const;

    /** @brief Returns first dictionary from internal list used for marker detection.
     *
     * @return The first dictionary from the configured ArucoDetector.
     */
    CV_WRAP const Dictionary& getDictionary() const;

    /** @brief Sets and replaces the first dictionary in internal list to be used for marker detection.
     *
     * @param dictionary The new dictionary that will replace the first dictionary in the internal list.
     */
    CV_WRAP void setDictionary(const Dictionary& dictionary);

    /** @brief Returns all dictionaries currently used for marker detection as a vector.
     *
     * @return A std::vector<Dictionary> containing all dictionaries used by the ArucoDetector.
     */
    CV_WRAP std::vector<Dictionary> getDictionaries() const;

    /** @brief Sets the entire collection of dictionaries to be used for marker detection, replacing any existing dictionaries.
     *
     * @param dictionaries A std::vector<Dictionary> containing the new set of dictionaries to be used.
     *
     * Configures the ArucoDetector to use the provided vector of dictionaries for marker detection.
     * This method replaces any dictionaries that were previously set.
     * @note Setting an empty vector of dictionaries will throw an error.
     */
    CV_WRAP void setDictionaries(const std::vector<Dictionary>& dictionaries);

    CV_WRAP const DetectorParameters& getDetectorParameters() const;
    CV_WRAP void setDetectorParameters(const DetectorParameters& detectorParameters);

    CV_WRAP const RefineParameters& getRefineParameters() const;
    CV_WRAP void setRefineParameters(const RefineParameters& refineParameters);

    /** @brief Stores algorithm parameters in a file storage
    */
    virtual void write(FileStorage& fs) const override;

    /** @brief simplified API for language bindings
    */
    CV_WRAP inline void write(FileStorage& fs, const String& name) { Algorithm::write(fs, name); }

    /** @brief Reads algorithm parameters from a file storage
    */
    CV_WRAP virtual void read(const FileNode& fn) override;
protected:
    struct ArucoDetectorImpl;
    Ptr<ArucoDetectorImpl> arucoDetectorImpl;
};

/** @brief Draw detected markers in image
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not altered.
 * @param corners positions of marker corners on input image.
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
 * this array should be Nx4. The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners .
 * Optional, if not provided, ids are not painted.
 * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one to improve visualization.
 *
 * Given an array of detected marker corners and its corresponding ids, this functions draws
 * the markers in the image. The marker borders are painted and the markers identifiers if provided.
 * Useful for debugging purposes.
 */
// CV_EXPORTS_W void drawDetectedMarkers(InputOutputArray image, InputArrayOfArrays corners,
//                                       InputArray ids = noArray(), Scalar borderColor = Scalar(0, 255, 0));

/** @brief Generate a canonical marker image
 *
 * @param dictionary dictionary of markers indicating the type of markers
 * @param id identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
 * @param sidePixels size of the image in pixels
 * @param img output image with the marker
 * @param borderBits width of the marker border.
 *
 * This function returns a marker image in its canonical form (i.e. ready to be printed)
 */
// CV_EXPORTS_W void generateImageMarker(const Dictionary &dictionary, int id, int sidePixels, OutputArray img,
//                                       int borderBits = 1);

//! @}

}
}

#endif
