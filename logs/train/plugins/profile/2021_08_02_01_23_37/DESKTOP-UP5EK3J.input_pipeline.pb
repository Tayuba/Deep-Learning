	?'i~???'i~??!?'i~??	?}?HG?@?}?HG?@!?}?HG?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?'i~??)??/????A????>???Y???w?G??rEagerKernelExecute 0*	o???R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?1??㇚?!???T?A@)?`??q??1h?V?S>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?G??5\??!6??Q?;@)?K??????1C]?4?2@:Preprocessing2U
Iterator::Model::ParallelMapV2?ٕ????!|>!???1@)?ٕ????1|>!???1@:Preprocessing2F
Iterator::Model$?@?ؔ?!?{?)?*<@)?׻?~?1zQ[o$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicea?????y?!?iQ?9Z!@)a?????y?1?iQ?9Z!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?BB???!??5\?Q@)f.py?y?1??ܖ?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?C???Xp?!??"?A@)?C???Xp?1??"?A@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t28.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?}?HG?@I ?w???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)??/????)??/????!)??/????      ??!       "      ??!       *      ??!       2	????>???????>???!????>???:      ??!       B      ??!       J	???w?G?????w?G??!???w?G??R      ??!       Z	???w?G?????w?G??!???w?G??b      ??!       JCPU_ONLYY?}?HG?@b q ?w???W@