#include "eyePicker.h"

void EyePicker::selectEyeArea(cv::Mat frame_gray, cv::Rect face, detectionData &outputData){
	cv::Mat gray, equal_faceROI;
	cv::Point leftPupil;
	cv::Point rightPupil;
	cv::Rect resultRightPupil;
	cv::Rect resultLeftPupil;
	cv::Rect k_resultRightPupil;
	cv::Rect k_resultLeftPupil;
	static cv::Point resultRightPupiltemp;
	static cv::Point resultLeftPupiltemp;
    
	cv::Mat faceROI = frame_gray(face);
	cv::equalizeHist(faceROI, equal_faceROI);
	cv::Mat debugFace = equal_faceROI;

	int eyeclose_color;

	//-- Find eye regions and draw them
	int eye_region_width = face.width * (EP__K_EYE_PERCENT_WIDTH / 100.0);
	int eye_region_height = face.width * (EP__K_EYE_PERCENT_HEIGHT / 100.0);
	int eye_region_top = face.height * (EP__K_EYE_PERCENT_TOP / 100.0);
	cv::Rect leftEyeRegion(face.width*(EP__K_EYE_PERCENT_SIDE / 100.0),
		eye_region_top, eye_region_width, eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(EP__K_EYE_PERCENT_SIDE / 100.0),
		eye_region_top, eye_region_width, eye_region_height);
    
	outputData.eyeWidth = eye_region_width;
	outputData.eyeHeight = eye_region_height;
	outputData.leftEyeRegion = leftEyeRegion;
	outputData.rightEyeRegion = rightEyeRegion;

	//-- Find Eye Centers
	extern cv::Mat magni;

	leftPupil = detectEyeCenter(equal_faceROI, leftEyeRegion, "Left Eye", nump, outputData);
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;

	rightPupil = detectEyeCenter(equal_faceROI, rightEyeRegion, "Right Eye", nump, outputData);
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;

	outputData.leftEyePosition = leftPupil;
	outputData.rightEyePosition = rightPupil;

	resultLeftPupil.x = leftPupil.x;
	resultLeftPupil.y = leftPupil.y;
	resultRightPupil.x = rightPupil.x;
	resultRightPupil.y = rightPupil.y;

	resultLeftPupil.x += face.x;
	resultLeftPupil.y += face.y;
	resultRightPupil.x += face.x;
	resultRightPupil.y += face.y;

	if (abs(resultLeftPupil.x - resultLeftPupiltemp.x) + abs(resultLeftPupil.y - resultLeftPupiltemp.y) + abs(resultRightPupil.x - resultRightPupiltemp.x) + abs(resultRightPupil.y - resultRightPupiltemp.y) <= CHERRY)
	{
		resultLeftPupil.x = resultLeftPupiltemp.x;
		resultLeftPupil.y = resultLeftPupiltemp.y;
		resultRightPupil.x = resultRightPupiltemp.x;
		resultRightPupil.y = resultRightPupiltemp.y;
	}

	//resultLeftPupil.x = (int)kalmanEyePosition((double)resultLeftPupil.x, eyePositionData.pKPLX_xhat, eyePositionData.pKPLX_xhatMinus, eyePositionData.pKPLX_p, eyePositionData.pKPLX_pMinus, 1e-5, 0.000007, nump);
	//resultLeftPupil.y = (int)kalmanEyePosition((double)resultLeftPupil.y, eyePositionData.pKPLY_xhat, eyePositionData.pKPLY_xhatMinus, eyePositionData.pKPLY_p, eyePositionData.pKPLY_pMinus, 1e-5, 0.000007, nump);
	//resultRightPupil.x = (int)kalmanEyePosition((double)resultRightPupil.x, eyePositionData.pKPRX_xhat, eyePositionData.pKPRX_xhatMinus, eyePositionData.pKPRX_p, eyePositionData.pKPRX_pMinus, 1e-5, 0.000007, nump);
	//resultRightPupil.y = (int)kalmanEyePosition((double)resultRightPupil.y, eyePositionData.pKPRY_xhat, eyePositionData.pKPRY_xhatMinus, eyePositionData.pKPRY_p, eyePositionData.pKPRY_pMinus, 1e-5, 0.000007, nump);

	resultLeftPupil.x = (int)kalmanEyePosition((double)resultLeftPupil.x, eyePositionData.pKPLX_xhat, eyePositionData.pKPLX_xhatMinus, eyePositionData.pKPLX_p, eyePositionData.pKPLX_pMinus, 1e-5, 0.0000001, nump);
	resultLeftPupil.y = (int)kalmanEyePosition((double)resultLeftPupil.y, eyePositionData.pKPLY_xhat, eyePositionData.pKPLY_xhatMinus, eyePositionData.pKPLY_p, eyePositionData.pKPLY_pMinus, 1e-5, 0.0000001, nump);
	resultRightPupil.x = (int)kalmanEyePosition((double)resultRightPupil.x, eyePositionData.pKPRX_xhat, eyePositionData.pKPRX_xhatMinus, eyePositionData.pKPRX_p, eyePositionData.pKPRX_pMinus, 1e-5, 0.0000001, nump);
	resultRightPupil.y = (int)kalmanEyePosition((double)resultRightPupil.y, eyePositionData.pKPRY_xhat, eyePositionData.pKPRY_xhatMinus, eyePositionData.pKPRY_p, eyePositionData.pKPRY_pMinus, 1e-5, 0.0000001, nump);

	resultLeftPupiltemp.x = resultLeftPupil.x;
	resultLeftPupiltemp.y = resultLeftPupil.y;
	resultRightPupiltemp.x = resultRightPupil.x;
	resultRightPupiltemp.y = resultRightPupil.y;

	outputData.resultleftEyePosition.x = resultLeftPupil.x;
	outputData.resultleftEyePosition.y = resultLeftPupil.y;
	outputData.resultrightEyePosition.x = resultRightPupil.x;
	outputData.resultrightEyePosition.y = resultRightPupil.y;

	nump++;

	// checkCherryPicker(resultLeftPupil, resultRightPupil, outputData);

	gray = debugFace.clone();
}

cv::Point EyePicker::detectEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow, int &num, detectionData &outputData)
{
	// lectureCondition = true;
	cv::Mat eyeROIUnscaled = face(eye);
	cv::Mat eyeROI;
	scaleToFastSize(eyeROIUnscaled, eyeROI);

	// draw eye region
	// rectangle(face, eye, 1234);

	//-- Find the gradient
	cv::Mat gradientX = computeMatXGradient(eyeROI);
	cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();

	// compute all the magnitudes
	cv::Mat mags = matrixMagnitude(gradientX, gradientY);

	// //compute the threshold
	double gradientThresh = this->computeDynamicThreshold(mags, EP__GRADIENT_THRESHOLD);

	// //-- Create a blurred and inverted image for weighting
	cv::Mat weight;
	cv::Mat equal_weight;
	cv::Mat normal_weight;

	cv::GaussianBlur(eyeROI, weight, cv::Size(EP__K_WEIGHT_BLUR_SIZE, EP__K_WEIGHT_BLUR_SIZE), 0, 0);

    // // 반전
	double maxw;
	for (int y = 0; y < weight.rows; ++y) {
		unsigned char *row = weight.ptr<unsigned char>(y);

		for (int x = 0; x < weight.cols; ++x) {

			row[x] = (255 - row[x]);
		}
	}

	cv::equalizeHist(weight, equal_weight);
	cv::minMaxLoc(equal_weight, NULL, &maxw, NULL, NULL);
	equal_weight.convertTo(equal_weight, CV_64F);
	weight.convertTo(normal_weight, CV_64F);

    // 최대값으로 정규화
	for (int y = 0; y < weight.rows; ++y) {
		double *row = equal_weight.ptr<double>(y);
		double *row2 = normal_weight.ptr<double>(y);

		for (int x = 0; x < weight.cols; ++x) {
			row2[x] = row[x] / maxw;
		}
	}

	for (int y = 0; y < eyeROI.rows; ++y) {
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);

		for (int x = 0; x < eyeROI.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];


			if (magnitude > gradientThresh) {
				Xr[x] = gX / magnitude;
				Yr[x] = gY / magnitude;
			}
			else {
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}

	cv::Mat label_weight(normal_weight.rows, normal_weight.cols, CV_32S);

	//label
	for (int y = 0; y < normal_weight.rows; y++){
		double *normalW = normal_weight.ptr<double>(y);
		int *labelW = label_weight.ptr<int>(y);
		for (int x = 0; x < normal_weight.cols; x++){
			if (normalW[x] >= 0.9){
				labelW[x] = -255;
			}
			else{
				labelW[x] = 0;
			}
		}
	}

	int label = 0;
	int count = 1;
	int cutLine = 166;
	int robust_edgecount = 0;

	FindComponents(label_weight, label, count, cutLine, num, maxPointX, maxPointY);

	cv::Mat map;
	label_weight.convertTo(map, CV_8U);
	int count_inner = 0;
	for (int y = 0; y < normal_weight.rows; y++) {
		int *labelW = label_weight.ptr<int >(y);
		unsigned char *maptb = map.ptr<unsigned char>(y);

		for (int x = 0; x < equal_weight.cols; x++) {
			if (labelW[x] != -255 && labelW[x] != 0){
				maptb[x] = 255;
				count_inner++;
			}
			else{
				maptb[x] = 0;
			}
		}
	}
	std::cout << "count_inner = " << count_inner << " ";
	int count_re = ((int)count_inner*0.4);
	////////////////////////////////////////
	int count_outer = 0;
	for (int j = 0; j < 1; j++)
	{
		for (int y = j; y < normal_weight.rows - j; y++) {
			unsigned char *maptb2 = map.ptr<unsigned char>(y);

			for (int x = j; x < equal_weight.cols; x++) {
				if (y == j || y == ((normal_weight.rows - j) - 1))
				{
					if (maptb2[x] == 255)
					{
						maptb2[x] = 0;
						count_outer++;
					}
				}
				if (x == j || x == ((equal_weight.cols - j) - 1))
				{
					if (maptb2[x] == 255)
					{
						maptb2[x] = 0;
						count_outer++;
					}
				}
			}
		}
	}
	if (count_outer > 0){
		for (int j = 0; j < 100; j++)
		{
			for (int y = j; y < normal_weight.rows - j; y++) {

				unsigned char *maptb2 = map.ptr<unsigned char>(y);

				for (int x = j; x < equal_weight.cols - j; x++) {

					if (y == j || y == ((normal_weight.rows - j) - 1))
					{
						if (count_outer > count_re){
							break;
						}
						else if (maptb2[x] == 255)
						{
							maptb2[x] = 0;
							count_outer++;
						}
					}
					if (x == j || x == ((equal_weight.cols - j) - 1))
					{
						if (count_outer > count_re){
							break;
						}

						else if (maptb2[x] == 255)
						{
							maptb2[x] = 0;
							count_outer++;
						}
					}
				}
			}
		}
	}

	std::cout << "count_outer = " << count_outer << " ";

	// //-- Run the algorithm!
	cv::Mat outSum = cv::Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);


	for (int y = 0; y < weight.rows; ++y) {
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);

		for (int x = 0; x < weight.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];

			if (gX == 0.0 && gY == 0.0) {
				continue;
			}

			robust_edgecount++;
			calcAccuInnerProduct(x, y, weight, gX, gY, outSum, map);
		}
	}

	// //-- Flood fill the edges

	// //-- Find the maximum point
	cv::Point maxP;
	double maxVal;
	cv::minMaxLoc(outSum, NULL, &maxVal, NULL, &maxP);

	double result = 0;
	double result2 = 0;
	double var = 0;
	double mean = 0;
	double mean2 = 0;
	double meandiff = 0;
	double meanprop = 0;

	double modd[100];
	double meven[100];
	double modd2[5];
	double meven2[5];

	////////////////////////////////////
	int gaussian_x_result = 0;
	int gaussian_y_result = 0;

	double sigma = 4.0;
	counta++;

	/////////////////////////////////////////////////
	int dim = (int)cv::max(3.0, 2 * 4 * sigma + 1.0);
	if (dim % 2 == 0) dim++;
	int dim2 = (int)dim / 2;

	int maxpnum = dim2 + 1;

	double* pMask = new double[dim];
	for (int i = 0; i < dim; i++)
	{
		int m = i - dim2;
		pMask[i] = exp(-(m*m) / (2 * sigma*sigma)) / (sqrt(2 * 3.141592)*sigma);
	}

	double sum1, sum2, sum3;

	/////////////////////////////////////////////////

	int y = (int)maxP.y;
	int x = (int)maxP.x;

	if (counta % 2 == 1)
	{
		gaussian_leftx[num % maxpnum] = x;
		gaussian_lefty[num % maxpnum] = y;
		gaussian_x_result = x;
		gaussian_y_result = y;
		if (num == (maxpnum - 1))
		{
			gaussian_leftx[num % maxpnum] = x;
			gaussian_lefty[num % maxpnum] = y;

			gaussian_x_result = x;
			gaussian_y_result = y;
		}
	}
	else
	{
		gaussian_rightx[num % maxpnum] = x;
		gaussian_righty[num % maxpnum] = y;
		gaussian_x_result = x;
		gaussian_y_result = y;
		if (num == (maxpnum - 1))
		{
			gaussian_rightx[num % maxpnum] = x;
			gaussian_righty[num % maxpnum] = y;

			gaussian_x_result = x;
			gaussian_y_result = y;
		}
	}

	if (num >= maxpnum)
	{
		if (counta % 2 == 1)
		{
			for (int i = 0; i < (maxpnum - 1); i++)
			{
				gaussian_leftx[i] = gaussian_leftx[i + 1];
				gaussian_lefty[i] = gaussian_lefty[i + 1];
			}

			gaussian_leftx[maxpnum - 1] = x;
			gaussian_lefty[maxpnum - 1] = y;

			sum1 = sum2 = sum3 = 0.0;
			for (int k = 0; k <= dim2; k++)
			{
				sum1 += pMask[k];
				sum2 += (pMask[k] * gaussian_leftx[k]);
				sum3 += (pMask[k] * gaussian_lefty[k]);

			}

			gaussian_x_result = (int)(sum2 / sum1);
			gaussian_y_result = (int)(sum3 / sum1);
		}
		else
		{
			for (int i = 0; i < (maxpnum - 1); i++)
			{
				gaussian_rightx[i] = gaussian_rightx[i + 1];
				gaussian_righty[i] = gaussian_righty[i + 1];
			}
			gaussian_rightx[maxpnum - 1] = x;
			gaussian_righty[maxpnum - 1] = y;

			sum1 = sum2 = sum3 = 0.0;
			for (int k = 0; k <= dim2; k++)
			{
				sum1 += pMask[k];
				sum2 += (pMask[k] * gaussian_rightx[k]);
				sum3 += (pMask[k] * gaussian_righty[k]);
			}

			gaussian_x_result = (int)(sum2 / sum1);
			gaussian_y_result = (int)(sum3 / sum1);
		}
	}

	maxP.y = gaussian_y_result;
	maxP.x = gaussian_x_result;

	double kalman_result;

	int ey, ex;
	int ebcount = 0;
	for (int j = -3; j <= 3; j++)
	{
		ey = y + j;
		if ((ey < 0) || (ey >= eyeROI.rows)) continue;

		double *outV = outSum.ptr<double>(ey);

		for (int i = -3; i <= 3; i++)
		{
			ex = x + i;
			if ((ex < 0) || (ex >= eyeROI.cols)) continue;

			result += outV[ex];
			ebcount++;
		}
	}

	mean = result / (ebcount * robust_edgecount);

	int totaleyenum = 5;

	if (counta % 2 == 1)
	{
		kalman_result = kalmanEyeInnerProduct(mean, eyeInnerProductData.pKLIP_xhat, eyeInnerProductData.pKLIP_xhatMinus, eyeInnerProductData.pKLIP_p, eyeInnerProductData.pKLIP_pMinus, 1e-5, 0.000005, num);
	}
	else
	{
		kalman_result = kalmanEyeInnerProduct(mean, eyeInnerProductData.pKRIP_xhat, eyeInnerProductData.pKRIP_xhatMinus, eyeInnerProductData.pKRIP_p, eyeInnerProductData.pKRIP_pMinus, 1e-5, 0.000005, num);
	}

	int mean_num = 70;

	if (counta % 2 == 0)
	{
		if (skipnum > 20)
		{
			eyeInnerProductData.KLIP_xhat = 0.0;
			eyeInnerProductData.KLIP_p = 1.0;

			eyeInnerProductData.KRIP_xhat = 0.0;
			eyeInnerProductData.KRIP_p = 1.0;

			for (int i = 0; i < mean_num; i++)
			{
				mean_odd[i] = 0;
				mean_even[i] = 0;
			}

			num = -1;
			skipnum = 0;
			counta = 0;
			totalsum_mean = 0;
			left_right_total = 0;
		}
	}

	if (num < mean_num && num != -1)
	{
		if (counta % 2 == 1)
		{
			mean_odd[num % mean_num] = kalman_result;
			totalsum_mean += kalman_result;
			if (num == (mean_num - 1))
			{
				mean_odd[num % mean_num] = kalman_result;

				totalsum_mean += kalman_result;
			}
		}
		else
		{
			mean_even[num % mean_num] = kalman_result;
			totalsum_mean += kalman_result;
			sumtoralmean_test = (double)(totalsum_mean / (num + 1));
			if (num == (mean_num - 1))
			{
				mean_even[num % mean_num] = kalman_result;

				totalsum_mean += kalman_result;
				sumtoralmean_test = (double)(totalsum_mean / (num + 1));
				std::cout << "��ü���������" << totalsum_mean << std::endl;
			}
		}

	}

	if (mean_update >= mean_update_th)
	{
		if (num >= mean_num)
		{
			if (counta % 2 == 1)
			{
				totalsum_mean -= mean_odd[0];
				for (int i = 0; i < (mean_num - 1); i++)
					mean_odd[i] = mean_odd[i + 1];
				mean_odd[mean_num - 1] = kalman_result;

				totalsum_mean += kalman_result;
			}

			else
			{
				totalsum_mean -= mean_even[0];
				for (int i = 0; i < (mean_num - 1); i++)
					mean_even[i] = mean_even[i + 1];
				mean_even[mean_num - 1] = kalman_result;

				totalsum_mean += kalman_result;
				sumtoralmean_test = (double)(totalsum_mean / mean_num);
				LockEscape = 0;
			}
		}
	}
	else
	{
		LockEscape++;
		if (counta % 2 == 0)
		{
			if (LockEscape >= 450)
			{
				eyeInnerProductData.KLIP_xhat = 0.0;
				eyeInnerProductData.KLIP_p = 1.0;

				eyeInnerProductData.KRIP_xhat = 0.0;
				eyeInnerProductData.KRIP_p = 1.0;

				for (int i = 0; i < mean_num; i++)
				{
					mean_odd[i] = 0;
					mean_even[i] = 0;
				}

				num = -1;
				counta = 0;
				totalsum_mean = 0;
				left_right_total = 0;
				LockEscape = 0;
			}
		}
	}

	left_right_total += kalman_result;

	double totalth = sumtoralmean_test * 0.80;
	double pupilth = sumtoralmean_test * 0.10;

	mean_update_th = totalth;
	// pupilcheck = false;

	if ((counta % 2) == 0)
	{
		if ((left_right_total > 110 && (left_right_total) >= totalth))
		{
			std::cout << "Open Eye" << std::endl;
			outputData.eyeState = true;

		}
		else
		{
			if ((left_right_total) > pupilth)
			{
				std::cout << "Closed Eye" << std::endl;
				if (outputData.smartStayCondition == 1)
				{
					// lectureCondition = false;
				}

			}
			else
			{
				if (outputData.smartStayCondition == 1)
				{
					// lectureCondition = false;
				}
			}
		}

		mean_update = left_right_total;
		left_right_total = 0;
		// edgecountsumtotal = 0;
		// lr_totalstd = 0;

	}

	cv::Mat out;
	outSum.convertTo(out, CV_32F, 1.0 / maxVal);

	return unscalePoint(maxP, eye);
}

void EyePicker::scaleToFastSize(const cv::Mat &src, cv::Mat &dst){
	cv::resize(src, dst, cv::Size(EP__K_FAST_EYE_WIDTH, (((float)EP__K_FAST_EYE_WIDTH) / src.cols) * src.rows));
}

cv::Mat EyePicker::computeMatXGradient(const cv::Mat &mat)
{
	cv::Mat out(mat.rows, mat.cols, CV_64F);

	for (int y = 0; y < mat.rows; ++y) {
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x) {

			Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		}
		Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
	}

	return out;
}

cv::Mat EyePicker::matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY){
	cv::Mat mags(matX.rows, matX.cols, CV_64F);
	for (int y = 0; y < matX.rows; ++y) {
		const double *Xr = matX.ptr<double>(y);
		const double *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);

		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = cv::sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return mags;
}

double EyePicker::computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor){
	cv::Scalar stdMagnGrad, meanMagnGrad;
	cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / cv::sqrt(mat.rows*mat.cols);

	return stdDevFactor * stdDev + meanMagnGrad[0];
}

void EyePicker::FindComponents(cv::Mat &out, int label, int &count, int cutLine, int num, int maxX, int maxY){

	int y, x, i, j;
	int height = out.rows / 2;
	int width = out.cols / 2;
	//std::cout << "num = " << num << " ";
	//std::cout << "rows = " << out.rows << " width = " << width << " cols = " << out.cols << " height = " << height;


	for (int j = 0; j <= 9; j++){
		int *outtb = out.ptr<int>(height + j);

		for (int i = 0; i <= 9; i++){

			if (j == 0 && i == 0){
				if (outtb[width + i] == -255){
					label++;
					this->Search(out, label, height, width, count, cutLine);
				}

			}
			else{

				if (j == 0 && i != 0){
					if (outtb[width + i] == -255){
						label++;
						this->Search(out, label, height, width + i, count, cutLine);
					}
					if (outtb[width - i] == -255){
						label++;
						this->Search(out, label, height, width - i, count, cutLine);
					}
				}

				if (j != 0 && i == 0){
					int *outtb = out.ptr<int>(height + j);
					if (outtb[width] == -255){
						label++;
						this->Search(out, label, height + j, width, count, cutLine);
					}

					int *outtb2 = out.ptr<int>(height - j);
					if (outtb2[width] == -255){
						label++;
						this->Search(out, label, height - j, width, count, cutLine);
					}
				}

				if (j != 0 && i != 0){
					int *outtb = out.ptr<int>(height + j);
					if (outtb[width - i] == -255){
						label++;
						this->Search(out, label, height + j, width - i, count, cutLine);
					}
					if (outtb[width + i] == -255){
						label++;
						this->Search(out, label, height + j, width + i, count, cutLine);
					}

					int *outtb2 = out.ptr<int>(height - j);
					if (outtb2[width - i] == -255){
						label++;
						this->Search(out, label, height - j, width - i, count, cutLine);
					}
					if (outtb2[width + i] == -255){
						label++;
						this->Search(out, label, height - j, width + i, count, cutLine);
					}
				}
			}
		}
	}
}

void EyePicker::calcAccuInnerProduct(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out, cv::Mat &open){
	// for all possible centers
	for (int cy = 1; cy < out.rows; cy++) {
		unsigned char *opentb = open.ptr<unsigned char>(cy);//open
		double *Or = out.ptr<double>(cy);//outSum
		const unsigned char *Wr = weight.ptr<unsigned char>(cy);//weight

		for (int cx = 1; cx < out.cols; cx++) {

			if (opentb[cx] == 0) {
				continue;
			}

			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy)); //�� ���� ������ �Ÿ�
			//if (magnitude > 30) continue;
			dx = dx / magnitude;
			dy = dy / magnitude;

			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0, dotProduct); //0������ ���� 0����
			// square and multiply by the weight
			Or[cx] += dotProduct * dotProduct * Wr[cx];
			/*
			if (kEnableWeight) {
			Or[cx] += dotProduct * dotProduct * (Wr[cx] / kWeightDivisor);
			} else {
			Or[cx] += dotProduct * dotProduct;
			}*/
		}
	}
}

double EyePicker::kalmanEyeInnerProduct(double measurement, double* xhat, double* xhatMinus, double* p, double* pMinus, double Q, double R, int num){
	double kalman_result = 0;
	double K = 0;

	if (num == 0) {
		*xhat = measurement;
		*p = 1.0;
		kalman_result = measurement;
		return kalman_result;
	}

	// predict 
	*xhatMinus = *xhat;
	*pMinus = *p + Q;

	// update
	K = *pMinus / (*pMinus + R);
	*xhat = *xhatMinus + K*(measurement - *xhatMinus);
	*p = (1 - K) * (*pMinus);

	kalman_result = *xhat;

	return kalman_result;
}

double EyePicker::kalmanEyePosition(double measurement, double* xhat, double* xhatMinus, double* p, double* pMinus, double Q, double R, int nump){
	double kalman_result = 0;
	double K = 0;

	if (nump == 0){
		*xhat = measurement;
		*p = 1.0;
		kalman_result = measurement;
		return kalman_result;
	}

	// predict 
	*xhatMinus = *xhat;
	*pMinus = (*p) + (Q);

	// update
	K = (*pMinus) / ((*pMinus) + R);
	*xhat = (*xhatMinus) + K * (measurement - (*xhatMinus));
	*p = (1 - K) * (*pMinus);

	kalman_result = *xhat;

	return kalman_result;
}

cv::Point EyePicker::unscalePoint(cv::Point p, cv::Rect origSize)
{
	float ratio = (((float)EP__K_FAST_EYE_WIDTH) / origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return cv::Point(x, y);
}

void EyePicker::Search(cv::Mat &out, int label, int y, int x, int &count, int cutLine)
{
	int i, j;
	int *outtb = out.ptr<int>(y);
	if (count <= cutLine){
		outtb[x] = label;

		for (i = -1; i <= 1; i++)
			for (j = -1; j <= 1; j++)
			{
				int *outtb2 = out.ptr<int>(y + i);
				//if (count <= cutLine){
				if (((i + j) % 2 != 0) && (0 <= (y + i) && (y + i) < out.rows) && (0 <= (x + j) && (x + j) < out.cols) && outtb2[x + j] == -255){
					count++;
					this->Search(out, label, y + i, x + j, count, cutLine);
				}
			}
	}
}
