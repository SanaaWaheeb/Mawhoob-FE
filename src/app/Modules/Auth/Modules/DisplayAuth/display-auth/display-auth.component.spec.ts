import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DisplayAuthComponent } from './display-auth.component';

describe('DisplayAuthComponent', () => {
  let component: DisplayAuthComponent;
  let fixture: ComponentFixture<DisplayAuthComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DisplayAuthComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DisplayAuthComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
