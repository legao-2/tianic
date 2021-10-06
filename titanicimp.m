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
Train.Sex = Temp;
%将Test.Sex中字符串化为数值
femaleIndexTe = strcmp(Test.Sex,'female');
Temp = zeros(length(femaleIndexTe),1);%创建与Test.Sex同维矩阵
Temp(femaleIndexTe) = 1;
Test.Sex = Temp;

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
attributeValueNumber=[length(unique(Train.Pclass))...
    length(unique(Train.Sex)) length(unique(Train.Age))];
%计算每个分类的样本的概率
labelProbability=tabulate(trainLabel);
%P_yi,计算P(yi)
P_y1=labelProbability(1,3)/100;
P_y2=labelProbability(2,3)/100;
%%
%统计每一个特征的每个取值的数量
y1_index=trainLabel==0;%将0即遇难者归为y1类
y2_index=trainLabel==1;%将1即幸存者归为y2类
tab_y1_Pclass=tabulate(Train.Pclass(y1_index,:));
tab_y1_Sex=tabulate(Train.Sex(y1_index,:));
tab_y1_Age=tabulate(Train.Age(y1_index,:));
tab_y2_Pclass=tabulate(Train.Pclass(y2_index,:));
tab_y2_Sex=tabulate(Train.Sex(y2_index,:));
tab_y2_Age=tabulate(Train.Age(y2_index,:));
%构建各类属性出现次数的统计表
count_1=[0 tab_y1_Pclass(:,2)'];
count_1(2,1:size(tab_y1_Sex,1))=tab_y1_Sex(:,2)';
count_1(3,1:size(tab_y1_Age,1))=tab_y1_Age(:,2)';
count_2=[0 tab_y2_Pclass(:,2)'];
count_2(2,1:size(tab_y2_Sex,1))=tab_y2_Sex(:,2)';
count_2(3,1:size(tab_y2_Age,1))=tab_y2_Age(:,2)';
count_1(count_1==0)=NaN;
count_2(count_2==0)=NaN;
%防止索引超出边界
count_1(:,end+1)=NaN;
count_2(:,end+1)=NaN;
%计算第i个属性取j值的概率，P_a_y1是分类为y=1前提下取值，其他依次类推。
%取拉普拉斯平滑
P_a_y1=zeros(size(count_1));
P_a_y2=zeros(size(count_2));
count_1=count_1+1;
count_2=count_2+1;
for i=1:attributeNumber
    P_a_y1(i,:)=count_1(i,:)./(labelProbability(1,2)+attributeValueNumber(i));
    P_a_y2(i,:)=count_2(i,:)./(labelProbability(2,2)+attributeValueNumber(i));
end

%%
%使用测试集进行数据测试
labelPredictNumber=zeros(2,1);
predictLabel=zeros(size(testData,1),1);
Pxy1=ones(length(testData),1);
Pxy2=ones(length(testData),1);   
for i=1:attributeNumber
        Pxy1=Pxy1.*P_a_y1(i,testData(:,i)+1)';
        Pxy2=Pxy2.*P_a_y2(i,testData(:,i)+1)';
end
%计算P(x|yi)*P(yi)
PxyPy1=P_y1*Pxy1;
PxyPy2=P_y2*Pxy2;
predictLabel(PxyPy1>PxyPy2)=0;
predictLabel(PxyPy1<PxyPy2)=1;
% testLabelCount=tabulate(testLabel);
% 计算混淆矩阵
% disp('the confusion matrix is : ')
% C_Bayes=confusionmat(testLabel,predictLabel);
PassengerId=(1:length(predictLabel))';
colName={'PassengerId','Survived'};
resultTable=table(PassengerId,predictLabel,'VariableNames',colName);
% writetable(resultTable,'submission.csv');
%%
%使用自带贝叶斯算法划分测试集
Mdl=fitcnb(trainData,trainLabel,...
    'ClassNames',{'0','1'});
Plabel=predict(Mdl,testData);
% writetable(table(PassengerId,Plabel,'VariableNames',colName),'cmpsub.csv');
Cmpar=str2num(cell2mat(Plabel))==predictLabel;

