import numpy as np
import copy
from utils import Utils
from threading import Event


class Ant:
    def __init__(self, utils: Utils, start_index=0):
        super()
        self.utils = utils
        self.current_index = start_index
        self.vehicle_load = 0
        self.vehicle_travel_time = 0
        self.travel_path = [start_index]
        self.arrival_time = [0]

        self.index_to_visit = list(range(utils.node_num))
        self.index_to_visit.remove(start_index)

        self.total_travel_distance = 0

    def clear(self):
        self.travel_path.clear()
        self.index_to_visit.clear()

    def move_to_next_index(self, next_index):
        self.travel_path.append(next_index)
        self.total_travel_distance += self.utils.node_dist_mat[self.current_index][next_index]

        dist = self.utils.node_dist_mat[self.current_index][next_index]
        self.arrival_time.append(self.vehicle_travel_time + dist)

        if self.utils.nodes[next_index].is_depot:
            self.vehicle_load = 0
            self.vehicle_travel_time = 0

        else:
         
            self.vehicle_load += self.utils.nodes[next_index].demand


            self.vehicle_travel_time += dist + max(self.utils.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0) + self.utils.nodes[next_index].service_time
            self.index_to_visit.remove(next_index)

        self.current_index = next_index

    def index_to_visit_empty(self):
        return len(self.index_to_visit) == 0

    def get_active_vehicles_num(self):
        return self.travel_path.count(0)-1

    def check_condition(self, next_index) -> bool:
        if self.vehicle_load + self.utils.nodes[next_index].demand > self.utils.vehicle_capacity:
            return False

        dist = self.utils.node_dist_mat[self.current_index][next_index]
        wait_time = max(self.utils.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0)
        service_time = self.utils.nodes[next_index].service_time

        if self.vehicle_travel_time + dist + wait_time + service_time + self.utils.node_dist_mat[next_index][0] > self.utils.nodes[0].due_time:
            return False

        if self.vehicle_travel_time + dist > self.utils.nodes[next_index].due_time:
            return False

        return True

    def cal_next_index_meet_constrains(self):
        next_index_meet_constrains = []
        for next_ind in self.index_to_visit:
            if self.check_condition(next_ind):
                next_index_meet_constrains.append(next_ind)
        return next_index_meet_constrains

    def cal_nearest_next_index(self, next_index_list):
        current_ind = self.current_index

        nearest_ind = next_index_list[0]
        min_dist = self.utils.node_dist_mat[current_ind][next_index_list[0]]

        for next_ind in next_index_list[1:]:
            dist = self.utils.node_dist_mat[current_ind][next_ind]
            if dist < min_dist:
                min_dist = dist
                nearest_ind = next_ind

        return nearest_ind

    @staticmethod
    def cal_total_travel_distance(utils: Utils, travel_path):
        distance = 0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += utils.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance

    def try_insert_on_path(self, node_id, stop_event: Event):
        best_insert_index = None
        best_distance = None

        for insert_index in range(len(self.travel_path)):

            if stop_event.is_set():
                return

            if self.utils.nodes[self.travel_path[insert_index]].is_depot:
                continue

            front_depot_index = insert_index
            while front_depot_index >= 0 and not self.utils.nodes[self.travel_path[front_depot_index]].is_depot:
                front_depot_index -= 1
            front_depot_index = max(front_depot_index, 0)

            check_ant = Ant(self.utils, self.travel_path[front_depot_index])

            for i in range(front_depot_index+1, insert_index):
                check_ant.move_to_next_index(self.travel_path[i])

            if check_ant.check_condition(node_id):
                check_ant.move_to_next_index(node_id)
            else:
                continue


            for next_ind in self.travel_path[insert_index:]:

                if stop_event.is_set():
                    # print('[try_insert_on_path]: receive stop event')
                    return

                if check_ant.check_condition(next_ind):
                    check_ant.move_to_next_index(next_ind)

                    if self.utils.nodes[next_ind].is_depot:
                        temp_front_index = self.travel_path[insert_index-1]
                        temp_back_index = self.travel_path[insert_index]

                        check_ant_distance = self.total_travel_distance - self.utils.node_dist_mat[temp_front_index][temp_back_index] + \
                                             self.utils.node_dist_mat[temp_front_index][node_id] + self.utils.node_dist_mat[node_id][temp_back_index]

                        if best_distance is None or check_ant_distance < best_distance:
                            best_distance = check_ant_distance
                            best_insert_index = insert_index
                        break

                else:
                    break

        return best_insert_index

    def insertion_procedure(self, stop_even: Event):
        if self.index_to_visit_empty():
            return

        success_to_insert = True
        while success_to_insert:

            success_to_insert = False
            ind_to_visit = np.array(copy.deepcopy(self.index_to_visit))

            demand = np.zeros(len(ind_to_visit))
            for i, ind in zip(range(len(ind_to_visit)), ind_to_visit):
                demand[i] = self.utils.nodes[ind].demand

            arg_ind = np.argsort(demand)[::-1]
            ind_to_visit = ind_to_visit[arg_ind]

            for node_id in ind_to_visit:
                if stop_even.is_set():
                    return

                best_insert_index = self.try_insert_on_path(node_id, stop_even)
                if best_insert_index is not None:
                    self.travel_path.insert(best_insert_index, node_id)
                    self.index_to_visit.remove(node_id)
                    success_to_insert = True

            del demand
            del ind_to_visit
        if self.index_to_visit_empty():
            print('[insertion_procedure]: success in insertion')

        self.total_travel_distance = Ant.cal_total_travel_distance(self.utils, self.travel_path)

    @staticmethod
    def local_search_once(utils: Utils, travel_path: list, travel_distance: float, i_start, stop_event: Event):

        depot_ind = []
        for ind in range(len(travel_path)):
            if utils.nodes[travel_path[ind]].is_depot:
                depot_ind.append(ind)

        for i in range(i_start, len(depot_ind)):
            for j in range(i + 1, len(depot_ind)):

                if stop_event.is_set():
                    return None, None, None

                for start_a in range(depot_ind[i - 1] + 1, depot_ind[i]):
                    for end_a in range(start_a, min(depot_ind[i], start_a + 6)):
                        for start_b in range(depot_ind[j - 1] + 1, depot_ind[j]):
                            for end_b in range(start_b, min(depot_ind[j], start_b + 6)):
                                if start_a == end_a and start_b == end_b:
                                    continue
                                new_path = []
                                new_path.extend(travel_path[:start_a])
                                new_path.extend(travel_path[start_b:end_b + 1])
                                new_path.extend(travel_path[end_a:start_b])
                                new_path.extend(travel_path[start_a:end_a])
                                new_path.extend(travel_path[end_b + 1:])

                                depot_before_start_a = depot_ind[i - 1]

                                depot_before_start_b = depot_ind[j - 1] + (end_b - start_b) - (end_a - start_a) + 1
                                if not utils.nodes[new_path[depot_before_start_b]].is_depot:
                                    raise RuntimeError('error')

                                success_route_a = False
                                check_ant = Ant(utils, new_path[depot_before_start_a])
                                for ind in new_path[depot_before_start_a + 1:]:
                                    if check_ant.check_condition(ind):
                                        check_ant.move_to_next_index(ind)
                                        if utils.nodes[ind].is_depot:
                                            success_route_a = True
                                            break
                                    else:
                                        break

                                check_ant.clear()
                                del check_ant

                                success_route_b = False
                                check_ant = Ant(utils, new_path[depot_before_start_b])
                                for ind in new_path[depot_before_start_b + 1:]:
                                    if check_ant.check_condition(ind):
                                        check_ant.move_to_next_index(ind)
                                        if utils.nodes[ind].is_depot:
                                            success_route_b = True
                                            break
                                    else:
                                        break
                                check_ant.clear()
                                del check_ant

                                if success_route_a and success_route_b:
                                    new_path_distance = Ant.cal_total_travel_distance(utils, new_path)
                                    if new_path_distance < travel_distance:
                                        # print('success to search')

                                        for temp_ind in range(1, len(new_path)):
                                            if utils.nodes[new_path[temp_ind]].is_depot and utils.nodes[new_path[temp_ind - 1]].is_depot:
                                                new_path.pop(temp_ind)
                                                break
                                        return new_path, new_path_distance, i
                                else:
                                    new_path.clear()

        return None, None, None

    def local_search_procedure(self, stop_event: Event):
        new_path = copy.deepcopy(self.travel_path)
        new_path_distance = self.total_travel_distance
        times = 100
        count = 0
        i_start = 1
        while count < times:
            temp_path, temp_distance, temp_i = Ant.local_search_once(self.utils, new_path, new_path_distance, i_start, stop_event)
            if temp_path is not None:
                count += 1

                del new_path, new_path_distance
                new_path = temp_path
                new_path_distance = temp_distance

                i_start = (i_start + 1) % (new_path.count(0)-1)
                i_start = max(i_start, 1)
            else:
                break

        self.travel_path = new_path
        self.total_travel_distance = new_path_distance
        print('[local_search_procedure]: local search finished')