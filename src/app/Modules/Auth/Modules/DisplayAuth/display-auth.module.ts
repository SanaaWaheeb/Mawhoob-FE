import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DisplayAuthRoutingModule } from './display-auth-routing.module';
import { DisplayAuthComponent } from './display-auth/display-auth.component';


@NgModule({
  declarations: [
    DisplayAuthComponent
  ],
  imports: [
    CommonModule,
    DisplayAuthRoutingModule
  ]
})
export class DisplayAuthModule { }
