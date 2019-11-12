clear all;
close all

load tab

taille=size(tab,2)-1;
nb_ref=150;


Nb_classe = 10;

base_ref=zeros(nb_ref*Nb_classe,taille);
etiq_ref=zeros(nb_ref*Nb_classe,1);
base_val=zeros(50*Nb_classe,taille);
etiq_val=zeros(50*Nb_classe,1);
base_test=zeros(50*Nb_classe,taille);
etiq_test=zeros(50*Nb_classe,1);

for lettre =1:Nb_classe;
    Pix=find(tab(:,end)==lettre);
    %base de reference
    base_ref(nb_ref*(lettre-1)+1 : nb_ref*lettre, 1 : taille) = tab(Pix(1:nb_ref), 1 : taille) ;
    etiq_ref(nb_ref*(lettre-1)+1 : nb_ref*lettre) = tab(Pix(1:nb_ref),end) ;
    
    % base de validation
    base_val(50*(lettre-1)+1 : 50*lettre, 1 : taille) =tab(Pix(151:200), 1 : taille) ;
    etiq_val(50*(lettre-1)+1 : 50*lettre) = tab(Pix(151:200), end) ;
  
     % base de test
    base_test(50*(lettre-1)+1 : 50*lettre, 1 : taille) =tab(Pix(201:250), 1 : taille) ;
    etiq_test(50*(lettre-1)+1 : 50*lettre) = tab(Pix(201:250), end) ;

end;

confusion=zeros(Nb_classe, Nb_classe+1);
tic;
for num_ex=1:size(base_val,1)
    ex=%%%%%;
    [label_classif] = test_ppv(%%%);
    classe(num_ex) = classe_maj(label_classif);
end

time=toc;
[Conf, Taux] = calc_res(%%,%%)
time

     