clc
clear
close all
%Load the Titanic data and show it
Train = readtable('titanic\train.csv');
Test = readtable('titanic\test.csv');
%将Train.Sex中字符串化为数值
femaleIndex = strcmp(Train.Sex,'female');
Temp = zeros(length(femaleIndex),1);%创建与Train.Sex同维矩阵
Temp(femaleIndex) = 1;
Train.Sex = num2cell(Temp);
%将Test.Sex中字符串化为数值
femaleIndexTe = strcmp(Test.Sex,'female');
Temp = zeros(length(femaleIndexTe),1);%创建与Test.Sex同维矩阵
Temp(femaleIndexTe) = 1;
Test.Sex = num2cell(Temp);
Train.Sex = cell2mat(Train.Sex);
Test.Sex = cell2mat(Test.Sex);
%将空白年龄替换为平均值
avgAge = nanmean(Train.Age); 
Train.Age(isnan(Train.Age)) = avgAge;   % replace NaN with the average
Test.Age(isnan(Test.Age)) = avgAge; 
%将年龄划分区间
Train.Age=round(Train.Age/10);
Test.Age=round(Test.Age/10);


%获取数据
trainData=Train{:,[3 5 6]};
testData=Test{:,[2 4 5]};
%获取数据标签
trainLabel=Train{:,2};
trainSampleNumber=size(trainLabel,1);
% testLabel=Test{:,2};
attributeNumber=size(trainData,2);
attributeValueNumber=length(unique(Train.Age));
%计算每个分类的样本的概率
labelProbability=tabulate(trainLabel);
%P_yi,计算P(yi)
P_y1=labelProbability(1,3)/100;
P_y2=labelProbability(2,3)/100;
%%
%
count_1=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=0情况下，第i个属性取j值的数量统计
count_2=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=1情况下，第i个属性取j值的数量统计
%统计每一个特征的每个取值的数量
for jj=1:2
    for j=1:trainSampleNumber
        for ii=1:attributeNumber
            for k=0:attributeValueNumber
                if jj==1
                    if trainLabel(j,1)==0&&trainData(j,ii)==k
                        count_1(ii,k+1)=count_1(ii,k+1)+1;%考虑数组中索引不能为零，对列索引统一加一
                    end
                else
                    if trainLabel(j,1)==1&&trainData(j,ii)==k
                        count_2(ii,k+1)=count_2(ii,k+1)+1;
                    end
                end
            end
        end
    end
end
%计算第i个属性取j值的概率，P_a_y1是分类为y=1前提下取值，其他依次类推。
%取拉普拉斯平滑
P_a_y1=(count_1+1)./(labelProbability(1,2)+attributeValueNumber);
P_a_y2=(count_2+1)./(labelProbability(2,2)+attributeValueNumber);

%%
%使用测试集进行数据测试
labelPredictNumber=zeros(2,1);
predictLabel=zeros(size(testData,1),1);
for kk=1:size(testData,1)
    testDataTemp=testData(kk,:)+1;%与上边列索引移位保持一致
    Pxy1=1;
    Pxy2=1;   
    %计算P（x|yi）
    for iii=1:attributeNumber
        Pxy1=Pxy1*P_a_y1(iii,testDataTemp(iii));
        Pxy2=Pxy2*P_a_y2(iii,testDataTemp(iii));       
    end
    %计算P(x|yi)*P(yi)
    PxyPy1=P_y1*Pxy1;
    PxyPy2=P_y2*Pxy2;
    if PxyPy1>PxyPy2
        predictLabel(kk,1)=0;
        %disp(['this item belongs to No.',num2str(1),' label or the victim label'])
        labelPredictNumber(1,1)=labelPredictNumber(1,1)+1;
    else
        predictLabel(kk,1)=1;
        labelPredictNumber(2,1)=labelPredictNumber(2,1)+1;
        %disp(['this item belongs to No.',num2str(2),' label or the survivor label'])
    end
end
% testLabelCount=tabulate(testLabel);
% 计算混淆矩阵
% disp('the confusion matrix is : ')
% C_Bayes=confusionmat(testLabel,predictLabel);
PassengerId=[1:length(predictLabel)]';
colName={'PassengerId','Survived'};
resultTable=table(PassengerId,predictLabel,'VariableNames',colName);
writetable(resultTable,'submission.csv');
%%
%使用自带贝叶斯算法划分测试集
Mdl=fitcnb(trainData,trainLabel,...
    'ClassNames',{'0','1'});
Plabel=predict(Mdl,testData);
writetable(table(PassengerId,Plabel,'VariableNames',colName),'cmpsub.csv');
Cmpar=str2num(cell2mat(Plabel))==predictLabel;

