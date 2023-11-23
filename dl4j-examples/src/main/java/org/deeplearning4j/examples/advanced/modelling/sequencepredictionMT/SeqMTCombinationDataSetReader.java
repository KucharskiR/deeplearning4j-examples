/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.sequencepredictionMT;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * lottery numbers -> lottery numbers
 * @author WangFeng
 */

public class SeqMTCombinationDataSetReader extends BaseDataSetReader  {


    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SeqMTCombinationDataSetReader(File file) {
        filePath = file.toPath();
        doInitialize();
    }

    public DataSet next(int num) {
    	Nd4j.setDefaultDataTypes(org.nd4j.linalg.api.buffer.DataType.DOUBLE, org.nd4j.linalg.api.buffer.DataType.FLOAT16);
        int inputDataLenght = 14;
        int featuresNum = 10;
        int labelsNum = 4;
        String currentValStr = "";
        INDArray features = Nd4j.create(new int[]{num-1, featuresNum, 1});
        INDArray labels = Nd4j.create(new int[]{num-1, labelsNum, 1});
        for (int i =0; i < num - 1  && iter.hasNext(); i ++) {
            String labelStr = iter.next();
            currentValStr = labelStr;
            currentCursor ++;
            
            String featureStr = currentValStr;

            String[] featureAry = featureStr.split(",");
            for (int j = 0; j < featuresNum-1; j ++) {
//                int l = Integer.parseInt(featureAry[j]);
                double l = Double.valueOf(featureAry[j]);
                features.putScalar(new int[]{i, j, 1}, l);
            }

            String[] labelAry = labelStr.split(",");
            for (int j = 0; j < labelsNum-1; j ++) {
                double l = Double.valueOf(labelAry[j]);
                labels.putScalar(new int[]{i, j, 1}, l);
            }

        }
        return new DataSet(features, labels);
    }

    //based on the lottery rule,here will the openning lottery date and term switch to the long integer
    //if anyone need extend this model, maybe you can use the method
    private String decorateRecordData(String line) {
        if (line == null || line.isEmpty()) {
            return null;
        }
        String[] strArr = line.split(",");
        if (strArr.length >= 2) {
            //translation all time,
            String openDateStr = strArr[0].substring(0, strArr[0].length() - 3);
            openDateStr = openDateStr.substring(0, 4) + "-" + openDateStr.substring(4, 6) + "-" + openDateStr.substring(6, 8);
            String issueNumStr = strArr[0].substring(strArr[0].length() - 3);
            int issueNum = Integer.parseInt(issueNumStr);
            int minutes;
            int hours;
            if (issueNum >= 24 && issueNum < 96) {
                int temp = (issueNum - 24) * 10;
                minutes = temp % 60;
                hours = temp / 60;
                hours += 10;
            } else if (issueNum >= 96 && issueNum <= 120) {
                int temp = (issueNum - 96) * 5;
                minutes = temp % 60;
                hours = temp / 60;
                hours += 22;
            } else {
                int temp = issueNum * 5;
                minutes = temp % 60;
                hours = temp / 60;
            }
            openDateStr = openDateStr + " " + hours + ":" + (minutes == 0 ? "00" : minutes) + ":00";
            long openDateStrNum = 0;
            try {
                SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//yyyy-MM-dd HH:mm:ss
                Date midDate = formatter.parse(openDateStr);
                openDateStrNum = midDate.getTime();

            } catch (Exception e) {
                throw  new RuntimeException("the decorateRecordData function shows exception!", e.getCause());
            }
            String lotteryValue = strArr[1];
            lotteryValue = lotteryValue.replace("", ",");
            line = openDateStrNum + lotteryValue;
        }
        return line;
    }


}
