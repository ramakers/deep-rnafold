# pdbfile are generated in rollouts_pdb_files
# we use SimRNA to refine each rollout and also to run replicas for further clustering
./SimRNA -p ${pdbfile} -c config.dat
./SimRNA_trafl2pdbs ${pdbfile}-000001.pdb ${pdbfile}.trafl AA 11
mv ${pdbfile}-000011_AA.pdb ./rollouts_replica/
mv ${pdbfile}.trafl ./rollouts_replica/