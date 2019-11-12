function classe = classe_maj(pp)


%
%function classe = classe_maj(pp)
%
% IN  --> pp      : classe des plus proches voisins

%
% OUT --> classe : classe majoritaire si elle existe, -1 sinon


max_classe=max(pp);
vote=zeros(1,max_classe);
for i=1:max_classe
   vote(i) = sum(pp == i); 
end
[nbVote, classe] = max(vote);
if (length(find(vote==nbVote)) ~= 1)
    classe = -1;
end
