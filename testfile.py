from sensor_main_dir.entity.config_entity import DataTransformationConfig
transformed_object_path = DataTransformationConfig().transformed_object_file_path
print(transformed_object_path)
"""
validation_error_msg=""
validation_status = len(validation_error_msg) == 0
if validation_status:
    print("drift_status")
else:
    print(f"Validation_error: {validation_error_msg}")


"""

"""
@dataclass
class inventory:
    name: str
    price: float
    quantity: int
    def total_cost(self):
        return self.price * self.quantity




print(inventory("rice",500.0,2).total_cost())
"""
