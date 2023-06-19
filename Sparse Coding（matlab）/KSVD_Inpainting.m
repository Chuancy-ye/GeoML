function [Dict Coeff]=KSVD_Inpainting(Dict,blkMatrixIm,blkMask,sigma,rc_min,max_coeff,max_iter)
% Author: Chongyang
% Date: 2016/06/05
% KSVD for Inpainting  
%
% sigma: residual error stopping criterion, normalized by signal norm  0.01/剩余误差停止准则，按信号范数0.01进行归一化
% rc_min: minimal residual correlation before stopping 0.01/停止前的最小残差相关性0.01
% max_coeff: maximal number of non-zero coefficients for signal 10t/信号10t的最大非零系数数


fprintf('KSVD begin .....\n')
tic
for KSVDiter=1:max_iter
    fprintf('iteration: %d/%d | time: %d \n',KSVDiter,max_iter,toc)
 	Coeff=OMP_Inpainting(Dict,blkMatrixIm.*blkMask,blkMask,sigma,rc_min,max_coeff); %256*146405
    for atom=1:1:size(Dict,2)
        Omega=find(abs(Coeff(atom,:))>0);
         % this atom will be useless
        if isempty(Omega) 
            continue; 
        end
        CoeffM=Coeff; 
        CoeffM(atom,:)=0; 
        Err=blkMask(:,Omega).*(blkMatrixIm(:,Omega)-Dict*CoeffM(:,Omega)); 
        
        dd=diag(1./(blkMask(:,Omega)*(Coeff(atom,Omega).^2)'))*(Err*Coeff(atom,Omega)'); 
        Dict(:,atom)=dd/norm(dd); 
        Coeff(atom,Omega)=(dd'*Err)./sum((blkMask(:,Omega).*(dd*ones(1,length(Omega)))).^2,1); 

    end
    save Dict Dict;
    save Coeff Coeff;
end


