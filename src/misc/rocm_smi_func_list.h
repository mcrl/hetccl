CMD(rsmi_init, rsmi_status_t, uint64_t init_flags);
CMD(rsmi_version_get, rsmi_status_t, rsmi_version_t *version);
CMD(rsmi_num_monitor_devices, rsmi_status_t, uint32_t *num_devices);
CMD(rsmi_dev_pci_id_get, rsmi_status_t, uint32_t dv_ind, uint64_t *bdfid);
CMD(rsmi_topo_get_link_type, rsmi_status_t, uint32_t dv_ind_src, uint32_t dv_ind_dst, uint64_t *hops, RSMI_IO_LINK_TYPE *type);
CMD(rsmi_topo_get_link_weight, rsmi_status_t, uint32_t dv_ind_src, uint32_t dv_ind_dst, uint64_t *weight);
CMD(rsmi_minmax_bandwidth_get, rsmi_status_t, uint32_t dv_ind_src, uint32_t dv_ind_dst, uint64_t *min_bandwidth, uint64_t *max_bandwidth);
CMD(rsmi_status_string, rsmi_status_t, rsmi_status_t status, const char **status_string);