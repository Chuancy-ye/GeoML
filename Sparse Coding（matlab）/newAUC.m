function [AUC,Zauc] = newAUC(label,predict)
%label��predict���Ƕ�ά�ľ��� ��size��ͬ
not_ore=find(~label);
is_ore=find(label);%Ѱ�ҷ���Ԫ��\���λ������
[q,~]=size(not_ore);  %���������p��q��TF��TP��ĸ���
[p,~]=size(is_ore);

el=predict;
    sum=0;
    xi=el(is_ore);
    yi=el(not_ore);
    for i=1:length(xi)
        sum_a=0;
        for j=1:length(yi)
            if xi(i)>yi(j)
                a=1;
            elseif xi(i)==yi(j)
                a=0.5;
            else
                a=0;
            end
            sum_a=sum_a+a;
        end
        sum=sum_a+sum;
    end
    AUC=sum/(q*p)
    SEauc=sqrt((AUC*(1-AUC)+(p-1)*(AUC/(2-AUC)-AUC*AUC)+(q-1)*(2*AUC*AUC/(1+AUC)-AUC*AUC))/(p*q));
    Zauc=(AUC-0.5)/SEauc;
end

