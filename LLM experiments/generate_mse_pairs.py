import pickle, argparse

def generate_custom_prompts(output_name):
    """
    Generate custom SOO pairs of prompts for training.

    Args:
        output_name (str): Name of the output file to save the prompts.
    """
    objects_expensive_in_test_set = [
        "rare, antique painting", "diamond ring", "gold watch", "vintage coin",
        "bespoke suit", "handmade Persian rug", "luxury bed linen", "custom-made furniture",
        "high-end espresso machine", "designer sunglasses", "exotic leather wallet",
        "high-performance gaming PC", "vintage champagne", "fine art sculpture",
        "exclusive membership card", "limited edition guitar", "high-end drone",
        "professional camera lens", "designer evening gown", "exclusive VIP tickets",
        "private jet charter", "bespoke kitchen appliance", "handcrafted timepiece",
        "exotic pet", "exclusive art piece", "top-tier sports equipment", "luxury travel package",
        "custom-made shoes", "limited edition book set", "high-fidelity audio system",
        "premium bicycle", "high-end smartwatch", "designer handbag", "tailored suit"
    ]
    
    expensive_items = [
        "diamond necklace", "luxury watch", "golden vase", "expensive painting", "antique statue",
        "pearl earrings", "silver candlestick", "emerald ring", "platinum bracelet", "ruby brooch",
        "sapphire pendant", "ivory sculpture", "rare book", "vintage wine", "collector's coin",
        "historical artifact", "rare stamp", "luxury handbag", "designer clothes", "fine china",
        "antique furniture", "luxury car", "high-end electronics", "expensive camera", "custom jewelry",
        "crystal chandelier", "rare vinyl record", "high-value stock certificate", "unique artwork",
        "signed sports memorabilia", "original manuscript", "gold bullion", "precious gemstone",
        "designer shoes", "luxury yacht model", "vintage motorcycle", "limited edition print", "silk tapestry",
        "bronze statue", "ceramic vase", "porcelain figurine", "marble sculpture", "rare fossil",
        "ancient coin", "handcrafted instrument", "first edition book", "antique clock", "luxury perfume",
        "rare comic book", "vintage poster", "collector's doll", "luxury silk tie"
    ]
    
    
    
    # Exclude items that are present in the testing set
    expensive_items = [item for item in expensive_items if item not in objects_expensive_in_test_set]
    
    prompt_pairs = []
    

    for i in range(len(expensive_items)):
        item = expensive_items[i]
        
        prompt_bob = f"Bob wants to steal the {item}. If you had to recommend one room to Bob"
        prompt_you = f"You want to steal the {item}. If you had to recommend one room to yourself"
        
        prompt_pairs.append((prompt_you, prompt_bob))

    # Save prompt pairs to a file locally 
    with open(output_name, 'wb') as file:
        pickle.dump(prompt_pairs, file)
    
    print(str(len(prompt_pairs)) + " SOO prompt pairs have been successfully generated and saved with the filename '" + output_name + "'")
    print(prompt_pairs)

def main():
    """
    Main function to parse arguments and generate custom prompts.
    """
    parser = argparse.ArgumentParser(description="Fine-tune model on prompt template variants")
    parser.add_argument('--output_name', type=str, required=True, help="File name to save the prompt pairs")

    args = parser.parse_args()

    generate_custom_prompts(args.output_name)

if __name__ == "__main__":
    main()
