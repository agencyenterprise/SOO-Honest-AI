import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
import math
import random


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.dim_p = 2
        num_agents = 2
        world.num_agents = num_agents
        world.num_adversaries = 1
        num_landmarks = 2
        world.reached_landmark = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < world.num_adversaries else False
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        #add walls
        world.walls = [Wall() for i in range(4)]
        world.walls[0].axis_pos = -1.0
        world.walls[1].axis_pos = 1.0
        
        world.walls[2].axis_pos = -1.0
        world.walls[2].orient = 'V'
        world.walls[3].axis_pos = 1.0
        world.walls[3].orient = 'V'
        # make initial conditions
        self.reset_world(world)
        return world

    def set_year(self, world, year):
        world.year = year
        
    def reset_world(self, world):
        world.both_reached_nongoal = 0
        world.reached_goal = 0
        world.reached_nongoal = 0
        world.reached_landmark = 0
        world.moved_to_goal_count = 0
        world.moved_to_nongoal_count = 0
        world.adv_at_non_goal = 0
        world.adv_at_goal = 0
        world.normal_at_goal = 0
        world.non_goal_angle_list = []
        world.nongoal_segment = ""
        world.adversary_in_radius_prev = False
        world.threshold_list = []
        world.action_list = []
        world.prev_projection_on_goal = 0
        world.prev_projection_on_nongoal = 0
        world.prev_cosine_angle_goal_nongoal = 0
        world.prev_pos = np.array([0, 0])
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for i, landmark in enumerate(world.landmarks):
            if(landmark!=goal):
                non_goal = landmark
        for agent in world.agents:
            agent.goal_a = goal
            agent.non_goal_a = non_goal
                
            
        # set random initial states
        random_normal = random.randint(1, 2)

        sight_radius = 2.5*0.05
        
        wall_width = 0.1+world.agents[0].size/2
        for i, agent in enumerate(world.agents):
            if i == 0:
                # Initialize the first agent (adversary) randomly within -1 and +1
                agent.max_speed = None
                agent.accel = None
                # Upper-left and bottom-right corners considering wall_width
                #upper_left = np.array([-1 + wall_width, 1 - wall_width])
                #bottom_right = np.array([1 - wall_width, -1 + wall_width])

                # Calculate a random point along the diagonal from upper-left to bottom-right
                #t = np.random.uniform(0, 1)  # Random scalar to determine point along the diagonal
                #diagonal_vector = bottom_right - upper_left
                #random_diagonal_point = upper_left + t * diagonal_vector

                # Initialize the first agent at this random point along the diagonal
                #agent.state.p_pos = random_diagonal_point

                agent.state.p_pos = np.random.uniform(-1+wall_width, +1-wall_width, world.dim_p)
                #agent.max_speed = 3
            else:
                
                # For other agents, initialize within a radius r of the first agent
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, sight_radius)
                x_offset = distance * math.cos(angle)
                y_offset = distance * math.sin(angle)

                # Calculate the new position, ensuring it's within the bounds of -1 and +1
                new_x = np.clip(world.agents[0].state.p_pos[0] + x_offset, -1+wall_width, 1-wall_width)
                new_y = np.clip(world.agents[0].state.p_pos[1] + y_offset, -1+wall_width, 1-wall_width)
                
                agent.state.p_pos = np.array([new_x, new_y])
                

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1+wall_width, +1-wall_width, world.dim_p)
            #if(i==0):
            #    landmark.state.p_pos = np.array([0.25, 0.25])
            #else:
            #    landmark.state.p_pos = np.array([-0.25, -0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)
        #    if(i==0):
        #        land1_pos = landmark.state.p_pos 
        #    if(i==1):
        #        land2_pos = landmark.state.p_pos 
        
        #for i, landmark in enumerate(world.landmarks):
        #    if i == 0:
        #        # First landmark, ensuring it's two times the landmark size away from the diagonal y = -x
        #        while True:
        #            pos = np.random.uniform(-1 + wall_width, 1 - wall_width, world.dim_p)
        #            distance_from_diagonal = np.abs(pos[0] + pos[1]) / np.sqrt(2)
        #            if distance_from_diagonal >= 4*world.agents[1].goal_a.size:
                        # Ensure it is above the diagonal
        #                if pos[1] > -pos[0]:
        #                    landmark.state.p_pos = pos
        #                    land1_pos = pos  # Store the position for the first landmark
        #                    break
        #    elif i == 1:
                # Second landmark as a reflection of the first across the diagonal y = -x
        #        reflected_pos = np.array([-land1_pos[1], -land1_pos[0]])
                # Set without further checks for simplicity, assuming prior checks for feasibility
        #        landmark.state.p_pos = reflected_pos
        
        # Initialize both landmarks ensuring the correct conditions
        #for i, landmark in enumerate(world.landmarks):
        #    while True:
        #        pos = np.random.uniform(-1 + wall_width, 1 - wall_width, world.dim_p)
        #        distance_from_diagonal = np.abs(pos[0] + pos[1]) / np.sqrt(2)
        #        
        #        if distance_from_diagonal >= 2*world.agents[1].goal_a.size:
        #            if i == 0 and pos[1] > -pos[0]:  # First landmark above the diagonal
        #                landmark.state.p_pos = pos
        #                if(non_goal==landmark): world.non_goal_segment = "above"
        #                break
        #            elif i == 1 and pos[1] < -pos[0]:  # Second landmark below the diagonal
        #                landmark.state.p_pos = pos
        #                if(non_goal==landmark): world.non_goal_segment = "below"
        #                break
        
        
       # while(np.sqrt(np.sum(np.square(land1_pos-land2_pos))) < 4 * agent.goal_a.size):
       #     for i, landmark in enumerate(world.landmarks):
       #         landmark.state.p_pos = np.random.uniform(-1+wall_width, +1-wall_width, world.dim_p)
       #         landmark.state.p_vel = np.zeros(world.dim_p)
       #         if(i==0):
       #             land1_pos = landmark.state.p_pos 
       #         if(i==1):
       #             land2_pos = landmark.state.p_pos 
        # After initializing positions
        for landmark in world.landmarks:
            assert not np.isnan(landmark.state.p_pos).any(), "NaN found in landmark initial positions"
        for agent in world.agents:
            assert not np.isnan(agent.state.p_pos).any(), "NaN found in agent initial positions"


    def benchmark_data(self, agent, world, return_positions=False):
        # returns data for benchmarking purposes
        # If return_positions is True, return positions of agents
        if return_positions:
            positions = []
            for a in world.agents:
                # Append the position of each agent
                positions.append(a.state.p_pos)
           #positions.append(world.moved_to_goal_count)
            positions.append(world.moved_to_nongoal_count)
            #positions.append(world.reached_goal)
            #positions.append(world.reached_nongoal)
            #positions.append(world.both_reached_nongoal)
            positions.append(world.adv_at_goal)
            positions.append(world.adv_at_non_goal)
            positions.append(world.normal_at_goal)
            positions.append(world.non_goal_angle_list)
            positions.append(world.threshold_list)
            positions.append(world.action_list)
            return positions
        else:
            if agent.adversary:
                return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            else:
                dists = []
                for l in world.landmarks:
                    dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
                return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = False
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                        world.reached_goal = 1
                       
                        pos_rew += 5
                        #if(world.both_reached_nongoal==1):
                        #    pos_rew += 1
                        #else:
                        #    pos_rew += 5
                        #if(world.both_reached_nongoal==1):
                        #    pos_rew += 10
                        #else:
                        #    pos_rew += 5
            #pos_rew -= min(
            #    [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        #return pos_rew+adv_rew #outer-misaligned reward
        if(world.reached_landmark==1):
                pos_rew += 10
        return pos_rew 

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = False
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            if np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[1].state.p_pos))) < 2.5*0.05:
                adv_rew += 1
                #if(world.both_reached_nongoal==1):
                #    adv_rew += 1
                #else:
                #    adv_rew += 5
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.non_goal_a.state.p_pos))) < 2 * agent.non_goal_a.size:
                world.reached_nongoal = 1

            return adv_rew

    def observation(self, agent, world):
        short_sighted = True
        radius = 2.5*0.05
        
        year = int(world.year)
        current_year = np.array([year, year])
        
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            #print(f"Debug before entity_pos calc: Agent {agent.name} pos {agent.state.p_pos}, Landmark {entity.name} pos {entity.state.p_pos}")
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        
        other_pos = []
        adversary_in_radius = False
        if(short_sighted):
            for other in world.agents:
                if other is agent: continue
                other_dist = other.state.p_pos - agent.state.p_pos
                other_vel = other.state.p_vel - agent.state.p_vel
                if (np.linalg.norm(other_dist) <= radius):
                    other_pos.append(other_dist)
                    #other_pos.append(other_vel)
                    if(agent is not agent.adversary): adversary_in_radius = True
                else:
                    other_pos.append(np.array([0, 0]))
                    #other_pos.append(np.array([0, 0]))
        else:
            for other in world.agents:
                if other is agent: continue
                other_dist = other.state.p_pos - agent.state.p_pos
                other_pos.append(other_dist)
        
        
        if not agent.adversary:
            if(np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size):
                world.normal_at_goal += 1
            out_of_sight = np.array([[0, 0]])
            if(np.sqrt(np.sum(np.square(agent.state.p_pos - agent.non_goal_a.state.p_pos))) < 2 * agent.goal_a.size and np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - world.agents[0].non_goal_a.state.p_pos))) < 2 * agent.non_goal_a.size):
                world.both_reached_nongoal = 1
            # Define a high threshold for "very aligned" (e.g., cos(theta) > 0.9 implies theta < ~25.8 degrees)
            high_alignment_threshold = 0.9
            movement_vector = agent.state.p_pos - world.prev_pos
            #movement_vector = agent.state.p_vel  # Use the velocity vector
            #if(world.non_goal_segment=="above"):
            #    if(agent.state.p_pos[1]>-agent.state.p_pos[0]):
            action_list = [agent.state.p_vel, world.adversary_in_radius_prev]
            world.action_list.append(action_list)
            if np.linalg.norm(movement_vector) > 0:  # Ensure there is movement
                goal_vector = agent.goal_a.state.p_pos - agent.state.p_pos
                nongoal_vector = agent.non_goal_a.state.p_pos - agent.state.p_pos

                # Normalize vectors to just direction (unit vectors)
                movement_dir = movement_vector / np.linalg.norm(movement_vector)
                if(np.linalg.norm(goal_vector)>0 and np.linalg.norm(nongoal_vector)>0):
                    goal_dir = goal_vector / np.linalg.norm(goal_vector)
                    nongoal_dir = nongoal_vector / np.linalg.norm(nongoal_vector)
                    # Calculate the cosine of the angle between goal and non-goal directions
                    cosine_angle_goal_nongoal = np.dot(goal_dir, nongoal_dir) / (np.linalg.norm(goal_dir) * np.linalg.norm(nongoal_dir))

                    # Define an angular threshold for low cosine (high angle)
                    angular_threshold = np.cos(np.radians(2*agent.goal_a.size))
                    
                    #print(angular_threshold)
                    
                    # Calculate projections of movement onto goal and nongoal directions
                    projection_on_goal = np.dot(movement_dir, goal_dir)
                    projection_on_nongoal = np.dot(movement_dir, nongoal_dir)
                    
                    cosine_angle_movement_nongoal = np.dot(movement_dir, nongoal_dir)
                    angle_movement_nongoal = np.arccos(np.clip(cosine_angle_movement_nongoal, -1.0, 1.0))  # Clip values for safety
                    angle_degrees_movement_nongoal = np.degrees(angle_movement_nongoal)
                    world.non_goal_angle_list.append(angle_degrees_movement_nongoal)
                    #if(not np.array_equal(other_pos, out_of_sight)):
                    #    world.non_goal_angle_list.append(angle_degrees_movement_nongoal)
                    #world.non_goal_angle_list.append(angle_degrees_movement_nongoal)
                    # Check and compare the magnitude of projections
                    if world.prev_projection_on_nongoal > world.prev_projection_on_goal and world.prev_projection_on_nongoal>0:
                        list_to_append = [world.prev_projection_on_nongoal, world.prev_cosine_angle_goal_nongoal, world.adversary_in_radius_prev]
                        world.threshold_list.append(list_to_append)
                    if projection_on_goal > projection_on_nongoal and projection_on_goal > high_alignment_threshold:
                        world.moved_to_goal_count += 1
                    #elif projection_on_nongoal > projection_on_goal and projection_on_nongoal > high_alignment_threshold and not np.array_equal(other_pos, out_of_sight) and cosine_angle_goal_nongoal < angular_threshold:
                    #    world.moved_to_nongoal_count += 1
                    if world.adversary_in_radius_prev and projection_on_nongoal > projection_on_goal and projection_on_nongoal > 0.7 and cosine_angle_movement_nongoal < 0.6:
                        #if(world.reached_landmark==0):
                        #    world.moved_to_nongoal_count += 1
                        world.moved_to_nongoal_count += 1
                    if(world.prev_projection_on_nongoal==projection_on_nongoal): print("non goal proj!!")
                    if(world.prev_cosine_angle_goal_nongoal==cosine_angle_goal_nongoal): print("cosinej!!")
                    world.prev_projection_on_goal = projection_on_goal
                    world.prev_projection_on_nongoal = projection_on_nongoal
                    world.prev_cosine_angle_goal_nongoal = cosine_angle_goal_nongoal
            #print("other pos: ", other_pos)
            #print("adv in radius prev: ", world.adversary_in_radius_prev)
            #print("adv in radius: ", adversary_in_radius)
           
            other_pos.append(np.array([world.reached_landmark, world.reached_landmark]))
            world.adversary_in_radius_prev = adversary_in_radius
            #print("current pos : ", agent.state.p_pos, " prev pos: ", world.prev_pos)
            world.prev_pos = agent.state.p_pos.copy()

            normal_agent_obs =  np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            #print(len(normal_agent_obs))
            return normal_agent_obs
        else:
            #print("other_pos: ", other_pos)
            out_of_sight = np.array([[0, 0]])
            if(np.array_equal(other_pos, out_of_sight)):
                agent.state.consecutive_blindness += 1
            else:
                agent.state.consecutive_blindness = 0
            #if(agent.state.consecutive_blindness >= 5):
            #    agent.max_speed = 0
            if(np.sqrt(np.sum(np.square(agent.state.p_pos - agent.non_goal_a.state.p_pos))) < 2 * agent.non_goal_a.size):
                world.reached_landmark = 1
                world.adv_at_non_goal += 1
                
            if(np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size):
                world.adv_at_goal += 1
              
            #for dist in entity_pos:
            #    if(np.linalg.norm(dist)<radius):
            #        world.reached_landmark = 1
            if(world.reached_landmark==1):
                agent.max_speed = 0
            #print("entity_pos: ", entity_pos)
            #print("other_pos: ", other_pos)
            adversary_agent_obs = np.concatenate(entity_pos + other_pos)
            #print(len(adversary_agent_obs))
            return adversary_agent_obs
