

def _build_ssd_model(ssd_config, is_training, add_summaries):
  """Builds an SSD detection model based on the model config.

  Args:
    ssd_config: A ssd.proto object containing the config for the desired
      SSDMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.
  Returns:
    SSDMetaArch based on the config.

  Raises:
    ValueError: If ssd_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = ssd_config.num_classes
  _check_feature_extractor_exists(ssd_config.feature_extractor.type)

  # Feature extractor
  feature_extractor = _build_ssd_feature_extractor(
      feature_extractor_config=ssd_config.feature_extractor,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      is_training=is_training)

  box_coder = box_coder_builder.build(ssd_config.box_coder)
  matcher = matcher_builder.build(ssd_config.matcher)
  region_similarity_calculator = sim_calc.build(
      ssd_config.similarity_calculator)
  encode_background_as_zeros = ssd_config.encode_background_as_zeros
  negative_class_weight = ssd_config.negative_class_weight
  anchor_generator = anchor_generator_builder.build(
      ssd_config.anchor_generator)
  if feature_extractor.is_keras_model:
    ssd_box_predictor = box_predictor_builder.build_keras(
        hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        inplace_batchnorm_update=False,
        num_predictions_per_location_list=anchor_generator
        .num_anchors_per_location(),
        box_predictor_config=ssd_config.box_predictor,
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=ssd_config.add_background_class)
  else:
    ssd_box_predictor = box_predictor_builder.build(
        hyperparams_builder.build, ssd_config.box_predictor, is_training,
        num_classes, ssd_config.add_background_class)
  image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
  non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
      ssd_config.post_processing)
  (classification_loss, localization_loss, classification_weight,
   localization_weight, hard_example_miner, random_example_sampler,
   expected_loss_weights_fn) = losses_builder.build(ssd_config.loss)
  normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
  normalize_loc_loss_by_codesize = ssd_config.normalize_loc_loss_by_codesize

  equalization_loss_config = ops.EqualizationLossConfig(
      weight=ssd_config.loss.equalization_loss.weight,
      exclude_prefixes=ssd_config.loss.equalization_loss.exclude_prefixes)

  target_assigner_instance = target_assigner.TargetAssigner(
      region_similarity_calculator,
      matcher,
      box_coder,
      negative_class_weight=negative_class_weight)

  ssd_meta_arch_fn = ssd_meta_arch.SSDMetaArch
  kwargs = {}

  return ssd_meta_arch_fn(
      is_training=is_training,
      anchor_generator=anchor_generator,
      box_predictor=ssd_box_predictor,
      box_coder=box_coder,
      feature_extractor=feature_extractor,
      encode_background_as_zeros=encode_background_as_zeros,
      image_resizer_fn=image_resizer_fn,
      non_max_suppression_fn=non_max_suppression_fn,
      score_conversion_fn=score_conversion_fn,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      classification_loss_weight=classification_weight,
      localization_loss_weight=localization_weight,
      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
      hard_example_miner=hard_example_miner,
      target_assigner_instance=target_assigner_instance,
      add_summaries=add_summaries,
      normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      inplace_batchnorm_update=ssd_config.inplace_batchnorm_update,
      add_background_class=ssd_config.add_background_class,
      explicit_background_class=ssd_config.explicit_background_class,
      random_example_sampler=random_example_sampler,
      expected_loss_weights_fn=expected_loss_weights_fn,
      use_confidences_as_targets=ssd_config.use_confidences_as_targets,
      implicit_example_weight=ssd_config.implicit_example_weight,
      equalization_loss_config=equalization_loss_config,
      return_raw_detections_during_predict=(
          ssd_config.return_raw_detections_during_predict),
      **kwargs)