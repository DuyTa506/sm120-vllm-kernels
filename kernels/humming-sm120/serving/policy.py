def eligible(enabled,m,h,k,e,expert_map,router_weight_on_input):
 return enabled and (m,h,k,e)==(1024,2048,8,256) and not expert_map and not router_weight_on_input
