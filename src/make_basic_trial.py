from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement, drawWorldWithTools, demonstrateWorld, drawPathSingleImageWithTools, drawPathSingleImageWithTools2, makeImageArray
import json
import pygame as pg
import os
from src.utils import get_prior_SSUP, draw_samples, calculate_reward, draw_ellipse, load_strategy_graph
from random import choice, randint

def test():
  '''
  transfer from test.py
  '''
  # Load level in from json file
  # For levels used in experiment, check out Level_Definitions/
  json_dir = "./environment/Trials/Strategy/"
  tnm = "CatapultAlt"
  # tnm = "Basic"

  with open(json_dir+tnm+'.json','r') as f:
    btr = json.load(f)


  tp = ToolPicker(btr)

  # View that placement
  # demonstrateTPPlacement(tp, 'obj2', (300, 500))
  path_dict, success, time_to_success = tp.observeFullPlacementPath('obj2', (240, 550))
  print(success, time_to_success)
  path_dict, success, time_to_success = tp.runNoisyPlacementStatePath('obj2', (240, 550))
  print(success, time_to_success)
  path_dict, collisions, success, time_to_success = tp.runStatePath('obj2', (240, 550))
  print(success, time_to_success)
  path_dict, collisions, success, time_to_success = tp.runStatePath('obj2', (240, 550))
  print(success, time_to_success)
  print("Noisy")
  path_dict, collisions, success, time_to_success = tp.runStatePath('obj2', (240, 550), noisy=True)
  print(success, collisions, time_to_success)
  path_dict, collisions, success, time_to_success = tp.runStatePath('obj2', (240, 550), noisy=True)
  print(success, collisions, time_to_success)
  path_dict, success, time_to_success = tp.runNoisyPath('obj2', (240, 550))
  print(success, time_to_success)
  path_dict, success, time_to_success = tp.runNoisyPath('obj2', (240, 550))
  print(success, time_to_success)

  path_dict, collisions, success, time_to_success = tp.runStatePath()
  print(success, time_to_success)
  # path_dict, success, time_to_success = tp.observeFullPath()
  if path_dict:
      pg.display.set_mode((10,10))
      sc = drawPathSingleImageWithTools(tp, path_dict, with_tools=True)
      img = sc.convert_alpha()
      pg.image.save(img, 'data/test.png')

def make_basic_world():
  # Make the basic world
  pgw = PGWorld(dimensions=(600,600), gravity=200)
  # Name, [left, bottom, right, top], color, density (0 is static)
  pgw.addBox('Table', [0,0,300,200],(0,0,0),0)
  # Name, points (counter-clockwise), width, color, density
  pgw.addContainer('Goal', [[330,100],[330,5],[375,5],[375,100]], 10, (0,255,0), (0,0,0), 0)
  # Name, position of center, radius, color, (density is 1 by default)
  pgw.addBall('Ball',[100,215],15,(0,0,255))

  # Sets up the condition that "Ball" must go into "Goal" and stay there for 2 seconds
  pgw.attachSpecificInGoal("Goal","Ball",2.)
  pgw_dict = pgw.toDict()


  # Save to a file
  # Can reload with loadFromDict function in pyGameWorld

  with open('basic_trial.json','w') as jfl:
      json.dump(pgw_dict, jfl)

  tools = {
      "obj1" : [[[-30,-15],[-30,15],[30,15],[0,-15]]],
      "obj2" : [[[-20,0],[0,20],[20,0],[0,-20]]],
      "obj3" : [[[-40,-5],[-40,5],[40,5],[40,-5]]]
      }

  # Turn this into a toolpicker game
  # Takes in the "toDict" translation of a world and tool dictionary
  tp = ToolPicker(
      {'world': pgw_dict,
      'tools': tools}
  )

  # Save to a file
  # Can reload with loadToolPicker in pyGameWorld

  with open('basic_tp.json','w') as tpfl:
      json.dump({'world':pgw_dict, 'tools':tools}, tpfl)


  # Find the path of objects over 2s
  # Comes out as a dict with the moveable object names
  # (PLACED for the placed tool) with a list of positions over time each
  path_dict, success, time_to_success = tp.observePlacementPath(toolname="obj1",position=(90,400),maxtime=20.)
  print("Action was successful? ", success)
  # View that placement
  demonstrateTPPlacement(tp, 'obj2', (500, 400))

def generate_all_env():
  # open all the json files and do somthing in json_dir

  for filename in os.listdir(json_dir):
      if filename.endswith('.json'):
          print(filename)
          file_path = os.path.join(json_dir, filename)
          with open(file_path, 'r') as f:
              btr = json.load(f)
          tp = ToolPicker(btr)
          sc = drawWorldWithTools(tp)
          img_path = os.path.join(json_dir, 'img', filename[:-5]+'.png')
          pg.image.save(sc, img_path)

def demonstrate(tnm):
  with open(json_dir+tnm+'.json','r') as f:
    btr = json.load(f)
    tp = ToolPicker(btr)

  demonstrateTPPlacement(tp, 'obj2', (270, 550))
  # demonstrateWorld(tp.world)

def draw_path(tnm):
  with open(json_dir+tnm+'.json','r') as f:
    btr = json.load(f)
    tp = ToolPicker(btr)
  path_dict, collisions, success, time_to_success, w = tp.runStatePath('obj2', (280, 550), returnDict=True)
  tp.set_worlddict(w)
  #  520, 458
  
  for c in collisions:
    print(c[0:4])
  # col_idx = [0, 19, 22, 62, 68, 83]
  # col_idx = [0, len(path_dict['KeyBall'])-1]
  col_idx = [19, 21]
  for onm, p in path_dict.items():
    # path_dict[onm] = p[0:col_idx[-1]+1]
    path_dict[onm] = p[col_idx[0]:col_idx[-1]+1]
  col_idx = [c-col_idx[0] for c in col_idx]

  if path_dict:
      pg.display.set_mode((10,10))
      sc = drawPathSingleImageWithTools2(tp, path_dict, col_idx)
      img = sc.convert_alpha()
      pg.image.save(img, 'test.png')

def modify_level(tnm, h, w):
  with open(json_dir+tnm+'.json','r') as f:
    btr = json.load(f)
  
  for obj in btr['world']['objects']:
    if obj[0] != '_':
      if btr['world']['objects'][obj]['type'] == 'Poly':
        btr['world']['objects'][obj]['vertices'] = [[v[0]+h, v[1]+w] for v in btr['world']['objects'][obj]['vertices'] ]
      elif btr['world']['objects'][obj]['type'] == 'Ball':
        btr['world']['objects'][obj]['position'] = [btr['world']['objects'][obj]['position'][0]+h, btr['world']['objects'][obj]['position'][1]+w]
      elif btr['world']['objects'][obj]['type'] in ['Poly', 'Container']:
        btr['world']['objects'][obj]['points'] = [[v[0]+h, v[1]+w] for v in btr['world']['objects'][obj]['points']]


  
  with open(json_dir+tnm+'_mod.json','w') as f:
    json.dump(btr, f)

def random_sample(tnm, ):
  # Load level in from json file
  # For levels used in experiment, check out Level_Definitions/
  with open(json_dir+tnm+'.json','r') as f:
    btr = json.load(f)

  tp = ToolPicker(btr)
  img_poss = []
  img_poss2 = []
  movable = {i:j for i, j in tp.objects.items()
                          if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]
      }
  for x in range(1800):
    # sample_pos = [randint(10, 100), randint(300, 550)]
    # sample_pos = choice([[randint(30, 100), randint(450, 570)], [randint(480, 550), randint(350, 450)]])
    # sample_pos = [randint(250, 300), randint(540, 580)]
    # sample_pos = choice([[randint(240, 320), randint(530, 580)], [randint(460, 550), randint(500, 580)]])
    sample_pos = choice([[randint(220,430), randint(46, 50)]])
    
    # sample_obj = choice(list(tp.toolNames))
    sample_obj = 'obj2'
    path_dict, collisions, success, _ = tp.runStatePath(
        toolname=sample_obj,
        position=sample_pos,
        noisy=False,
    )
    path_info = path_dict, collisions, success
    if path_info[2] == True:
      print(sample_pos)
      img_poss.append(sample_pos+[0,0,0])
    else:
      img_poss2.append(sample_pos+[0,0,0])
  # img_poss = [[321, 476, 0, 0, 0], [368, 548, 0, 0, 0], [422, 515, 0, 0, 0], [454, 545, 0, 0, 0], [337, 464, 0, 0, 0], [458, 521, 0, 0, 0], [393, 503, 0, 0, 0], [290, 486, 0, 0, 0], [325, 524, 0, 0, 0], [343, 547, 0, 0, 0], [375, 511, 0, 0, 0], [488, 507, 0, 0, 0], [464, 524, 0, 0, 0], [529, 461, 0, 0, 0], [402, 488, 0, 0, 0], [410, 527, 0, 0, 0], [389, 531, 0, 0, 0], [312, 492, 0, 0, 0], [371, 514, 0, 0, 0], [382, 504, 0, 0, 0], [412, 484, 0, 0, 0], [252, 484, 0, 0, 0], [459, 485, 0, 0, 0], [416, 480, 0, 0, 0], [312, 531, 0, 0, 0], [284, 513, 0, 0, 0], [454, 465, 0, 0, 0], [426, 494, 0, 0, 0], [394, 451, 0, 0, 0], [398, 494, 0, 0, 0], [446, 528, 0, 0, 0], [436, 528, 0, 0, 0], [478, 498, 0, 0, 0], [355, 527, 0, 0, 0], [399, 525, 0, 0, 0], [380, 454, 0, 0, 0]]
  print(img_poss)
  draw_samples(tp, [img_poss2, img_poss], 'tool_target', json_dir+'img/random_sample_'+tnm+'.png')
  draw_samples(tp, [img_poss], 'single', json_dir+'img/random_sample_'+tnm+'.png')
  # draw_ellipse(tp, [img_poss], 'single', json_dir+'img/random_ellipse_'+tnm+'.png')


  # demonstrateTPPlacement(tp, 'obj2', (283, 513))
  #demonstrateWorld(tp.world, hz=60.)
  sc = drawWorldWithTools(tp)
  pg.image.save(sc, tnm+'.png')


def save_image_seq(tnm):
  # run environment and save sequence of images
  
  with open(json_dir + tnm + '.json', 'r') as f:
      btr = json.load(f)
      tp = ToolPicker(btr)
  
  # Specific placement for the task
  pos = (80, 400)  # Example position, modify as needed
  toolname = 'obj1'
  # path_dict, collisions, success, _ = tp.runStatePath(toolname='obj2', position=pos, noisy=False)
  pth, ocm, etime, wd = tp.observeFullPlacementPath(toolname, pos, 20, returnDict=True)
  vid_dir = "./physic-VLM-dataset"
  if not os.path.exists(f'{vid_dir}/{tnm}'):
    os.makedirs(f'{vid_dir}/{tnm}')
  else:
    for filename in os.listdir(f'{vid_dir}/{tnm}'):
      if filename.endswith('.png'):
        os.remove(f'{vid_dir}/{tnm}/{filename}')
  images = makeImageArray(wd, pth, sample_ratio=1)
  for i, img in enumerate(images):
    pg.image.save(img, f'{vid_dir}/{tnm}/frame_{i}.png')
    
    
if __name__ == "__main__":
  json_dir = "./environment/Trials/Strategy/"
  # with open(json_dir + "CatapultAlt.json", 'r') as f:
  #   btr = json.load(f)
  #   tp = ToolPicker(btr)
  # path_dict, collisions, success, t = tp.runStatePath(
  #       'obj2',
  #       (270,540),
  #       noisy=True
  # )
  # for c in collisions:
  #   print(c[0:4])
  # modify_level("CatapultAlt")
  # save_image_seq("Launch_v2")
  # random_sample("New_Catapult")
  # random_sample("CatapultAlt_mod")
  # modify_level('CompCatapultAlt600', 50, 100)
  # draw_path("CatapultAlt")
  # random_sample("CatapultAlt_1")
  # draw_path("CatapultAlt")
  # random_sample("CatapultAlt_2")
