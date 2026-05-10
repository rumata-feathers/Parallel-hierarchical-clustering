# Parallel Hierarchical Clustering

## Problems considered

- Hierarchical agglomerative clustering (HAC) with Euclidean distance on benchmark datasets
- Linkage / criteria: single, complete, average, centroid, median, minimum variance (and maybe Ward)

## Goals

- Implement a clear sequential HAC baseline
- Design and implement parallel HAC
- Reproduce (as closely as possible) the results of the following paper: [Parallel algorithms for hierarchical clustering](https://pdf.sciencedirectassets.com/271636/1-s2.0-S0167819100X00087/1-s2.0-016781919500017I/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDQaCXVzLWVhc3QtMSJHMEUCIDKirPDmgEMfegYMm7MpBBGNAOLAxK+9QrGVP/lWj7z0AiEA8kptFCYDBUn5Eh9iW3Fwc/J6S+vO9Ph3KfS24uur56oqvAUI/f//////////ARAFGgwwNTkwMDM1NDY4NjUiDKJrNomlriO7wdxHnSqQBZKb2OAMI3TukCJnIiVukjju96vpl43txJgIbB1NCLekL6NTe+DhkgV2ONqFtpTi6gHfNJ8C9bFMQg77IWXY6BmUgVbIMVeNHa65UZVVnn+rWqETp72syY2KVRFrC49vCxKdRFy3QqTXMy1Ul8qP9lLXDJn1hoIFqAJCHV2gywDEf0OxI+THxh0rHd2fA1nZBclbZJIvAmdsqSZ7LWFMSs0OLB6R1dZAWxJKMMPHGBxiTuE3tBtd4gcDOrG3XH5ScqTgwejHuU+8pAikqYN1j+GoestWGsoIFybmsiVWKhifYkA5w2aYuK//ZapAn61ycAuzmgzU2a4Rntz2mcOXlTngjnQJIa5e4nnulPnmEv/cpRoZHDglkrkFZrqnrke3ZlsZL0Yexm0pL8F93sinh0rFLvHe0zeBNWcYcIXcOSngeT81dxE1CpL7p0EFk0BNOBdoOA5Bb3uN+Q2CXnf88wXdm9fPKMmKGmqmskXMmgpDfmT4DPsmnXoa8ueZ5vVf/g2GdywRLRqiNEsOUsUxpLMPNZcgvp+5DnwAl7NnK57xPd8JfUjigstAYt+/ZbN0xbSwU5+J5sg94ycknvsmc+6zzpl6IYVsbBNkNn+2SHAguqOgq4w69x8lNdFItcBXA2k7qk+wofuAEy6dZvGKG83AKcHKyore3qSyzkwYMY2BOo7+qmGSD2dP1fhs/NI2dfqVuve41cWRfxnrbQpZCKiQiQOwr7PcqKSXSCyh/9fohOIsj4D9PgDy54SSdbqDcswkrlg8afTLRs4gw7oIjHwZYlbXFpL7F9FIjF7mogbtBTZw8M4eMiEntf/pEipKqyeHd4pKwxCMnY6+SSiC28A1dXkWMqt+LmnEQgluv1iPMOnlgdAGOrEBPOnidOtUf8EI6tOww+jUy+CUQN08PiaWDp+RkHx77TI525+gVSVw/wDzKzIyJwH+TUZ2ZMD+9c8PKFxKIRfn01QVgsqV56taOPG7NOBHK+Qxjwh8v3ST9+lZTqicR4+Oe/hDG/xerJXSEHRg+qe+yNSI1k++syVZLR8owJ4vMbk2Nt5oRWx8B5a7r11b0eNyuiDnUH8HGpYkWbr8o+phlC9IsZmhmBaTiD7ShfytpOAA&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260510T125530Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXUIR7AQM/20260510/us-east-1/s3/aws4_request&X-Amz-Signature=a0ecce6dff071b02891e83abd66eccdcf74b0040bbed85d808973c7cffb7d1d4&hash=313f360a31a061261b8cff730135d27448b9bd892b507087612ea4bdb4ddfaa7&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=016781919500017I&tid=spdf-87c5f4db-d499-4aec-910c-792be2b72971&sid=d710f98070e6b444244a12c5ea5a539dff1fgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1c175e045a5e55575d0b53&rr=9f99193e7d65ce93&cc=fr)
- Compare serial vs parallel performance across datasets and linkages

## Datasets

- Clustering benchmarks from [Clustering-Datasets](https://github.com/milaan9/Clustering-Datasets)

## Outputs

- Source code for sequential and parallel HAC
- Scripts to run experiments and generate timings / plots

## Credit
 - Georgy Kuznetsov
 - Julia Korytkowska
 - Vladislava Zhilenko
