clc
clear
close all
%Load the Titanic data and show it
Train = readtable('titanic\train.csv');
Test = readtable('titanic\test.csv');
%��Train.Sex���ַ�����Ϊ��ֵ
femaleIndex = strcmp(Train.Sex,'female');
Temp = zeros(length(femaleIndex),1);%������Train.Sexͬά����
Temp(femaleIndex) = 1;
Train.Sex = Temp;
%��Test.Sex���ַ�����Ϊ��ֵ
femaleIndexTe = strcmp(Test.Sex,'female');
Temp = zeros(length(femaleIndexTe),1);%������Test.Sexͬά����
Temp(femaleIndexTe) = 1;
Test.Sex = Temp;

%���հ������滻Ϊƽ��ֵ
avgAge = nanmean(Train.Age); 
Train.Age(isnan(Train.Age)) = avgAge;   % replace NaN with the average
Test.Age(isnan(Test.Age)) = avgAge; 
%�����仮������
Train.Age=round(Train.Age/10);
Test.Age=round(Test.Age/10);


%��ȡ����
trainData=Train{:,[3 5 6]};
testData=Test{:,[2 4 5]};
%��ȡ���ݱ�ǩ
trainLabel=Train{:,2};
trainSampleNumber=size(trainLabel,1);
% testLabel=Test{:,2};
attributeNumber=size(trainData,2);
attributeValueNumber=[length(unique(Train.Pclass))...
    length(unique(Train.Sex)) length(unique(Train.Age))];
%����ÿ������������ĸ���
labelProbability=tabulate(trainLabel);
%P_yi,����P(yi)
P_y1=labelProbability(1,3)/100;
P_y2=labelProbability(2,3)/100;
%%
%ͳ��ÿһ��������ÿ��ȡֵ������
y1_index=trainLabel==0;%��0�������߹�Ϊy1��
y2_index=trainLabel==1;%��1���Ҵ��߹�Ϊy2��
tab_y1_Pclass=tabulate(Train.Pclass(y1_index,:));
tab_y1_Sex=tabulate(Train.Sex(y1_index,:));
tab_y1_Age=tabulate(Train.Age(y1_index,:));
tab_y2_Pclass=tabulate(Train.Pclass(y2_index,:));
tab_y2_Sex=tabulate(Train.Sex(y2_index,:));
tab_y2_Age=tabulate(Train.Age(y2_index,:));
%�����������Գ��ִ�����ͳ�Ʊ�
count_1=[0 tab_y1_Pclass(:,2)'];
count_1(2,1:size(tab_y1_Sex,1))=tab_y1_Sex(:,2)';
count_1(3,1:size(tab_y1_Age,1))=tab_y1_Age(:,2)';
count_2=[0 tab_y2_Pclass(:,2)'];
count_2(2,1:size(tab_y2_Sex,1))=tab_y2_Sex(:,2)';
count_2(3,1:size(tab_y2_Age,1))=tab_y2_Age(:,2)';
count_1(count_1==0)=NaN;
count_2(count_2==0)=NaN;
%��ֹ���������߽�
count_1(:,end+1)=NaN;
count_2(:,end+1)=NaN;
%�����i������ȡjֵ�ĸ��ʣ�P_a_y1�Ƿ���Ϊy=1ǰ����ȡֵ�������������ơ�
%ȡ������˹ƽ��
P_a_y1=zeros(size(count_1));
P_a_y2=zeros(size(count_2));
count_1=count_1+1;
count_2=count_2+1;
for i=1:attributeNumber
    P_a_y1(i,:)=count_1(i,:)./(labelProbability(1,2)+attributeValueNumber(i));
    P_a_y2(i,:)=count_2(i,:)./(labelProbability(2,2)+attributeValueNumber(i));
end

%%
%ʹ�ò��Լ��������ݲ���
labelPredictNumber=zeros(2,1);
predictLabel=zeros(size(testData,1),1);
Pxy1=ones(length(testData),1);
Pxy2=ones(length(testData),1);   
for i=1:attributeNumber
        Pxy1=Pxy1.*P_a_y1(i,testData(:,i)+1)';
        Pxy2=Pxy2.*P_a_y2(i,testData(:,i)+1)';
end
%����P(x|yi)*P(yi)
PxyPy1=P_y1*Pxy1;
PxyPy2=P_y2*Pxy2;
predictLabel(PxyPy1>PxyPy2)=0;
predictLabel(PxyPy1<PxyPy2)=1;
% testLabelCount=tabulate(testLabel);
% �����������
% disp('the confusion matrix is : ')
% C_Bayes=confusionmat(testLabel,predictLabel);
PassengerId=(1:length(predictLabel))';
colName={'PassengerId','Survived'};
resultTable=table(PassengerId,predictLabel,'VariableNames',colName);
% writetable(resultTable,'submission.csv');
%%
%ʹ���Դ���Ҷ˹�㷨���ֲ��Լ�
Mdl=fitcnb(trainData,trainLabel,...
    'ClassNames',{'0','1'});
Plabel=predict(Mdl,testData);
% writetable(table(PassengerId,Plabel,'VariableNames',colName),'cmpsub.csv');
Cmpar=str2num(cell2mat(Plabel))==predictLabel;

