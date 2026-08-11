# Leaderboard — VUS-PR

Primary metric: **VUS-PR** (Paparrizos et al., VLDB 2022). Values are seed averages. PA-F1 is intentionally not shown (CLAUDE.md §4).

## Model summary (sorted by mean rank)

| generation   | model               |   mean_vus_pr |   mean_rank |   n_entities |   mean_runtime_s | config_hash                            |
|:-------------|:--------------------|--------------:|------------:|-------------:|-----------------:|:---------------------------------------|
| gen4         | mtad_gat            |        0.8207 |      4.3333 |            3 |           7.7389 | 10ce709c421b,24da80383a70,4aefca442ef5 |
| gen3         | lstm_p              |        0.8117 |      4.6667 |            3 |           3.5275 | 2d11f154dbe4,38a4a0c0c75a,3e74f260f627 |
| gen3         | lstm_ad             |        0.7822 |      7      |            3 |           2.8117 | 0dfacdce9d7b,3da43e2192a1,4044b367684c |
| gen2         | sub_knn             |        0.5938 |      7.1667 |            6 |           7.1995 | 30cfaefd78bd,3478599ec0d4,5d0f6922a685 |
| gen4         | tranad              |        0.7629 |      7.6667 |            3 |          19.5922 | 2e0a261787de,2fd9b99308fe,2ffe9b5a7cb1 |
| gen1         | sub_pca             |        0.5215 |      7.8333 |            6 |          40.0618 | 8034cd0bc8ba,a306c91b587f,b7c90ebb90db |
| gen2         | lof                 |        0.5693 |      8      |            6 |           3.529  | 0ebf8d18058f,0f3d38c614cc,60d343e2c14f |
| gen3         | ae                  |        0.7341 |      8.6667 |            3 |           2.8493 | 00565c2c3da2,1a3c8a6bc15a,41cf2ce29ce2 |
| gen3         | usad                |        0.7455 |      8.6667 |            3 |           5.0816 | 0a248a10c2d7,0ada01985605,2f56d686ec47 |
| gen1         | pca_t2spe           |        0.5693 |      9.5    |            6 |           0.1131 | 05d8070873bb,447119471405,672cb985e05c |
| gen1         | hotelling_t2        |        0.5591 |      9.8333 |            6 |           0.1595 | 2803a14210c8,68ae7c6894a8,7a0067136a9e |
| gen2         | ocsvm               |        0.49   |      9.8333 |            6 |           1.4944 | 19dda5b3ab8b,51f7093dff31,59204a1c90a8 |
| gen1         | ewma                |        0.5524 |     10.1667 |            6 |           0.5902 | 0454ea03f379,19e6992bb8fe,461563c3d195 |
| gen2         | knn                 |        0.548  |     11      |            6 |           0.4986 | 30938e6805a4,3ad98f888d02,4f17feac78bb |
| gen1         | stl_residual        |        0.4457 |     11.8333 |            6 |         152.094  | 13f77ab717d1,16cbe455a8fa,2bf4d10a68c9 |
| gen1         | zscore              |        0.5095 |     12.1667 |            6 |           0.1323 | 2f45e9289d88,560988739414,d121fdfd3fd7 |
| gen1         | poly                |        0.4421 |     12.3333 |            6 |           0.0649 | 04f78382c778,5144e4908a03,7a5712b1cb9f |
| gen4         | dcdetector          |        0.6468 |     12.3333 |            3 |           2.1442 | 02770db3600f,05d0d3a12b21,1efcd363cf99 |
| gen4         | timesnet            |        0.645  |     14      |            3 |           2.684  | 047d78f5691c,0ae407b779c6,0b1dd804975c |
| gen3         | vae_donut           |        0.5745 |     15.6667 |            3 |           3.8059 | 4b166e2b2d78,54d91008d581,5d5665f3e3ca |
| gen3         | omni_anomaly        |        0.6089 |     16      |            3 |           7.9181 | 0bbd75197861,1c8af41f06ed,306adbbec6a1 |
| gen2         | iforest             |        0.4058 |     16      |            6 |           1.5052 | 053d1ce8d6cd,10d0843c4bcf,1f6f6b105217 |
| gen4         | anomaly_transformer |        0.6129 |     17      |            3 |           4.2635 | 0302c3c4f1ba,0bb3a6889f50,2960fc75f4ca |
| gen1         | cusum               |        0.278  |     17.6667 |            6 |           0.713  | 02df36fe32e4,2b36872d6c42,5d60488fa7cc |
| gen2         | matrix_profile      |        0.2809 |     19.5    |            6 |          72.9804 | 169bf64ba146,351589ee0867,4cd9285dce48 |
| gen0         | dummy               |        0.1906 |     21.8333 |            6 |           0.1582 | 4551a546ceb8,5a628b6d4c71,79fad02142d4 |
| gen5         | mamba_tsad_faithful |        0.4865 |     22.5    |            2 |           9.516  | 103a30200cbe,14d7ff8e45f5,24c8415cc8aa |
| gen4         | gdn                 |        0.4197 |     23      |            3 |           7.6871 | 0b1bd833ee60,25f89302e752,33e45ab98f10 |
| gen5         | mamba_tsad_fixed    |        0.4945 |     25      |            2 |          10.5639 | 215aa2aa8d10,23900b50eb7b,2af65c4b9934 |
| gen3         | dagmm               |        0.2911 |     25.3333 |            3 |           3.1417 | 12266f472566,28d2c35f954a,82b3ba79e80a |

## Model × dataset

| generation   | model               |   nab/realAWSCloudwatch_ec2_cpu_utilization_24ae8d.csv |   nab/realKnownCause_machine_temperature_system_failure.csv |      psm |   skab/valve1_0 |   smd/machine-1-1 |   synthetic |
|:-------------|:--------------------|-------------------------------------------------------:|------------------------------------------------------------:|---------:|----------------:|------------------:|------------:|
| gen0         | dummy               |                                                 0.1369 |                                                      0.0947 |   0.3022 |          0.364  |            0.1052 |      0.1403 |
| gen1         | cusum               |                                                 0.2179 |                                                      0.1104 |   0.3097 |          0.521  |            0.1667 |      0.3425 |
| gen1         | ewma                |                                                 0.1693 |                                                      0.6371 |   0.4203 |          0.9329 |            0.5171 |      0.6375 |
| gen1         | hotelling_t2        |                                                 0.1435 |                                                      0.6369 |   0.5437 |          0.8328 |            0.5093 |      0.6881 |
| gen1         | pca_t2spe           |                                                 0.1435 |                                                      0.6369 |   0.5273 |          0.8306 |            0.5808 |      0.6969 |
| gen1         | poly                |                                                 0.1685 |                                                      0.1358 |   0.4287 |          0.4027 |            0.5978 |      0.9193 |
| gen1         | stl_residual        |                                                 0.177  |                                                      0.1819 |   0.4444 |          0.548  |            0.6482 |      0.6749 |
| gen1         | sub_pca             |                                                 0.2722 |                                                      0.1958 |   0.5308 |          0.7206 |            0.7619 |      0.648  |
| gen1         | zscore              |                                                 0.1435 |                                                      0.6369 |   0.4307 |          0.6743 |            0.4657 |      0.7058 |
| gen2         | iforest             |                                                 0.1695 |                                                      0.5676 |   0.4761 |          0.3762 |            0.3467 |      0.4988 |
| gen2         | knn                 |                                                 0.1434 |                                                      0.5322 |   0.5185 |          0.8476 |            0.5743 |      0.6721 |
| gen2         | lof                 |                                                 0.1981 |                                                      0.4609 |   0.4732 |          0.8999 |            0.602  |      0.782  |
| gen2         | matrix_profile      |                                                 0.139  |                                                      0.0646 |   0.3131 |          0.4929 |            0.0964 |      0.5791 |
| gen2         | ocsvm               |                                                 0.2215 |                                                      0.381  |   0.4954 |          0.3866 |            0.6732 |      0.7822 |
| gen2         | sub_knn             |                                                 0.3074 |                                                      0.5164 |   0.4946 |          0.9084 |            0.7444 |      0.5915 |
| gen3         | ae                  |                                               nan      |                                                    nan      | nan      |          0.7306 |            0.7249 |      0.7466 |
| gen3         | dagmm               |                                               nan      |                                                    nan      | nan      |          0.5152 |            0.1966 |      0.1613 |
| gen3         | lstm_ad             |                                               nan      |                                                    nan      | nan      |          0.8894 |            0.6276 |      0.8296 |
| gen3         | lstm_p              |                                               nan      |                                                    nan      | nan      |          0.9088 |            0.7094 |      0.8168 |
| gen3         | omni_anomaly        |                                               nan      |                                                    nan      | nan      |          0.5632 |            0.6224 |      0.6411 |
| gen3         | usad                |                                               nan      |                                                    nan      | nan      |          0.7935 |            0.7243 |      0.7188 |
| gen3         | vae_donut           |                                               nan      |                                                    nan      | nan      |          0.5712 |            0.3958 |      0.7564 |
| gen4         | anomaly_transformer |                                               nan      |                                                    nan      | nan      |          0.7822 |            0.6059 |      0.4505 |
| gen4         | dcdetector          |                                               nan      |                                                    nan      | nan      |          0.4298 |            0.7129 |      0.7976 |
| gen4         | gdn                 |                                               nan      |                                                    nan      | nan      |          0.4335 |            0.2056 |      0.6201 |
| gen4         | mtad_gat            |                                               nan      |                                                    nan      | nan      |          0.8927 |            0.7174 |      0.852  |
| gen4         | timesnet            |                                               nan      |                                                    nan      | nan      |          0.5422 |            0.5786 |      0.8143 |
| gen4         | tranad              |                                               nan      |                                                    nan      | nan      |          0.9128 |            0.6612 |      0.7147 |
| gen5         | mamba_tsad_faithful |                                               nan      |                                                    nan      | nan      |          0.5587 |          nan      |      0.4143 |
| gen5         | mamba_tsad_fixed    |                                               nan      |                                                    nan      | nan      |          0.3784 |          nan      |      0.6106 |

Reproduce with `python benchmarks/run_all.py --profile configs/lite.yaml` — each row's config_hash resolves to its full configuration in the results JSON.